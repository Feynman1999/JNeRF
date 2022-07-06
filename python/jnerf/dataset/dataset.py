
import random
import jittor as jt
from jittor.dataset import Dataset
import os
import json
import cv2
import imageio
from math import pi
from math import tan
from tqdm import tqdm
import numpy as np
from jnerf.utils.registry import DATASETS
import glob
from xml.dom.minidom import parse
NERF_SCALE = 0.33


def fov_to_focal_length(resolution: int, degrees: float):
	return 0.5*resolution/tan(0.5*degrees*pi/180)

def write_image_imageio(img_file, img, quality):
	img = (np.clip(img, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
	kwargs = {}
	if os.path.splitext(img_file)[1].lower() in [".jpg", ".jpeg"]:
		if img.ndim >= 3 and img.shape[2] > 3:
			img = img[:,:,:3]
		kwargs["quality"] = quality
		kwargs["subsampling"] = 0
	imageio.imwrite(img_file, img, **kwargs)

def read_image_imageio(img_file):
	img = imageio.imread(img_file)
	img = np.asarray(img).astype(np.float32)
	if len(img.shape) == 2:
		img = img[:,:,np.newaxis]
	return img / 255.0

def srgb_to_linear(img):
	limit = 0.04045
	return np.where(img > limit, np.power((img + 0.055) / 1.055, 2.4), img / 12.92)

def linear_to_srgb(img):
	limit = 0.0031308
	return np.where(img > limit, 1.055 * (img ** (1.0 / 2.4)) - 0.055, 12.92 * img)

def jt_srgb_to_linear(img):
	limit = 0.04045
	return jt.ternary(img > limit, jt.pow((img + 0.055) / 1.055, 2.4), img / 12.92)

def jt_linear_to_srgb(img):
	limit = 0.0031308
	return jt.ternary(img > limit, 1.055 * (img ** (1.0 / 2.4)) - 0.055, 12.92 * img)

def read_image(file):
	if os.path.splitext(file)[1] == ".bin":
		with open(file, "rb") as f:
			bytes = f.read()
			h, w = struct.unpack("ii", bytes[:8])
			img = np.frombuffer(bytes, dtype=np.float16, count=h*w*4, offset=8).astype(np.float32).reshape([h, w, 4])
	else:
		img = jt.array(read_image_imageio(file))
	return img.numpy()

def write_image(file, img, quality=95):
	if os.path.splitext(file)[1] == ".bin":
		if img.shape[2] < 4:
			img = np.dstack((img, np.ones([img.shape[0], img.shape[1], 4 - img.shape[2]])))
		with open(file, "wb") as f:
			f.write(struct.pack("ii", img.shape[0], img.shape[1]))
			f.write(img.astype(np.float16).tobytes())
	else:
		if img.shape[2] == 4:
			img = np.copy(img)
			# Unmultiply alpha
			img[...,0:3] = np.divide(img[...,0:3], img[...,3:4], out=np.zeros_like(img[...,0:3]), where=img[...,3:4] != 0)
			img[...,0:3] = linear_to_srgb(img[...,0:3])
		else:
			img = linear_to_srgb(img)
		write_image_imageio(file, img, quality)


@DATASETS.register_module()
class NerfDataset():
	def __init__(self,root_dir, para_path, batch_size, mode='train', H=2048, W=2592, aabb_scale=8.0, img_alpha=True,to_jt=True, have_img=True, preload_shuffle=True):
		self.root_dir=root_dir
		self.para_path = para_path
		self.batch_size=batch_size
		self.preload_shuffle=preload_shuffle
		self.H=H
		self.W=W
		self.aabb_scale = aabb_scale

		self.resolution=[0,0]# W*H
		self.transforms_gpu=[]
		self.metadata=[]
		self.image_data=[]
		self.focal_lengths=[]
		self.n_images=0
		self.img_alpha=img_alpha# img RGBA or RGB
		self.to_jt=to_jt
		self.have_img=have_img
		self.compacted_img_data=[]# img_id ,rgba,ray_d,ray_o
		assert mode=="train" or mode=="val"
		self.mode=mode
		self.idx_now=0
		self.load_data()
		jt.gc()
		self.image_data = self.image_data.reshape(
			self.n_images, -1, 4).detach()

	def __next__(self):
		if self.idx_now+self.batch_size >= self.shuffle_index.shape[0]:
			del self.shuffle_index
			self.shuffle_index=jt.randperm(self.n_images*self.H*self.W).detach()
			jt.gc()
			self.idx_now = 0      
		img_index=self.shuffle_index[self.idx_now:self.idx_now+self.batch_size]
		img_ids,rays_o,rays_d,rgb_target=self.generate_random_data(img_index,self.batch_size)
		self.idx_now+=self.batch_size
		return img_ids, rays_o, rays_d, rgb_target
		
	def load_data(self,root_dir=None):
		print(f"load {self.mode} data")
		if root_dir is None:
			root_dir=self.root_dir
		
		root_dir = root_dir.replace("\\", '/') # windows
		scene_name = root_dir.split("/")[-2]
		para_path = os.path.join(self.para_path, scene_name)
		image_path = sorted(glob.glob(root_dir +  "/*.jpg"))

		if self.mode=="val":
			image_path = image_path[::10]

		self.n_images = len(image_path)

		poses = []
		cam_ids = [] # int
		metadata = np.zeros((self.n_images, 11), dtype=np.float32)
		focal_lengths = np.zeros((self.n_images, 2), dtype=np.float32)
		now_idx = 0

		for frame in tqdm(image_path):
			x_ratio = 1.0
			y_ratio = 1.0
			if self.have_img:
				img_path = frame
				if not os.path.exists(img_path):
					raise RuntimeError(f"img: {img_path} not exists")
				img = read_image(img_path)
				if img.shape[0] != self.H or img.shape[1] != self.W:
					x_ratio = self.W * 1.0 / img.shape[1]
					y_ratio = self.H * 1.0 / img.shape[0]
					img = cv2.resize(img, (self.W, self.H), interpolation=cv2.INTER_AREA)
				self.image_data.append(img) # 
			else:
				self.image_data.append(np.zeros((self.H,self.W,3)))

			cam_id = frame.split(".")[-2]
			cam_ids.append(int(cam_id[3:5]))
			cam_id = str(int(cam_id[3:5]))

			pose_path = os.path.join(para_path, cam_id, 'extrinsics.xml')
			pose = get_pose_from_xml(pose_path) # ndarray 3*4
			self.transforms_gpu.append(pose)
			poses.append(pose)

			intrinsic_path = os.path.join(para_path, cam_id, 'intrinsic.xml')
			K, D = get_intrinsic_from_xml(intrinsic_path)
			# K: (3,3)  D: (5, ) [k1, k2, p1, p2, k3]
			
			# fix K if resize
			K[0, :] *= x_ratio
			K[1, :] *= y_ratio

			metadata[now_idx, 0] = D[0]
			metadata[now_idx, 1] = D[1]
			metadata[now_idx, 2] = D[2]
			metadata[now_idx, 3] = D[3]
			assert D[4] == 0
			metadata[now_idx, 4] = K[0, 2]
			metadata[now_idx, 5] = K[1, 2]
			metadata[now_idx, 6] = K[0, 0]
			metadata[now_idx, 7] = K[1, 1]
			metadata[now_idx, 8:] = np.array([0,0,0])

			focal_lengths[now_idx, 0] = metadata[now_idx, 6]
			focal_lengths[now_idx, 1] = metadata[now_idx, 7]

			now_idx += 1

	
		self.resolution=[self.W,self.H]
		self.resolution_gpu=jt.array(self.resolution)
		self.metadata=jt.array(metadata)

		aabb_range=(0.5,0.5)
		self.aabb_range=(aabb_range[0]-self.aabb_scale/2,aabb_range[1]+self.aabb_scale/2)

		self.H=int(self.H)
		self.W=int(self.W)

		self.image_data=jt.array(self.image_data)
		self.transforms_gpu=jt.array(self.transforms_gpu)
		self.focal_lengths=jt.array(focal_lengths)
		## transpose to adapt Eigen::Matrix memory
		self.transforms_gpu=self.transforms_gpu.transpose(0,2,1)
		
		if self.img_alpha and self.image_data.shape[-1]==3:
			self.image_data=jt.concat([self.image_data,jt.ones(self.image_data.shape[:-1]+(1,))],-1).stop_grad()
		self.shuffle_index=jt.randperm(self.H*self.W*self.n_images).detach()
		jt.gc()
	
	def generate_random_data(self,index,bs):
		img_id=index//(self.H*self.W)
		img_offset=index%(self.H*self.W)
		focal_length =self.focal_lengths[img_id]
		xforms = self.transforms_gpu[img_id]
		principal_point = self.metadata[:, 4:6][img_id]
		xforms=xforms.permute(0,2,1) # permute back
		rays_o = xforms[...,  3]
		x=((img_offset%self.W)+0.5)
		y=((img_offset//self.W)+0.5)
		xy=jt.stack([x,y],dim=-1)
		rays_d = jt.concat([(xy - principal_point) /focal_length, jt.ones([bs, 1])], dim=-1) # [bs, 3]
		rays_d = jt.normalize(xforms[ ...,  :3].matmul(rays_d.unsqueeze(2)))
		rays_d=rays_d.squeeze(-1)
		rgb_tar=self.image_data.reshape(-1,4)[index]
		return img_id,rays_o,rays_d,rgb_tar

	def generate_rays_total(self, img_id,H,W):
		H=int(H)
		W=int(W)
		img_size=H*W
		focal_length =self.focal_lengths[img_id]
		xforms = self.transforms_gpu[img_id]
		principal_point = self.metadata[:, 4:6][img_id]
		xy = jt.stack(jt.meshgrid((jt.linspace(0, H-1, H)+0.5)/H, (jt.linspace(0,
					  W-1, W)+0.5)/W), dim=-1).permute(1, 0, 2).reshape(-1, 2)
		# assert H==W
		# xy += (jt.rand_like(xy)-0.5)/H
		xforms=xforms.permute(1,0)
		rays_o = xforms[:,  3]
		res = jt.array(self.resolution)
		rays_d = jt.concat([(xy-principal_point)* res/focal_length, jt.ones([H*W, 1])], dim=-1)
		rays_d = jt.normalize(xforms[ :,  :3].matmul(rays_d.unsqueeze(2)))
		rays_d=rays_d.squeeze(-1)
		return rays_o, rays_d

	def generate_rays_total_test(self, img_ids, H, W):
		# select focal,trans,p_point
		focal_length = jt.gather(
			self.focal_lengths, 0, img_ids)
		xforms = jt.gather(self.transforms_gpu, 0, img_ids)
		principal_point = jt.gather(
			self.metadata[:, 4:6], 0, img_ids)
		# rand generate uv 0~1
		xy = jt.stack(jt.meshgrid((jt.linspace(0, H-1, H)+0.5)/H, (jt.linspace(0,
					  W-1, W)+0.5)/W), dim=-1).permute(1, 0, 2).reshape(-1, 2)
		# assert H==W
		# xy += (jt.rand_like(xy)-0.5)/H
		xy_int = jt.stack(jt.meshgrid(jt.linspace(
			0, H-1, H), jt.linspace(0, W-1, W)), dim=-1).permute(1, 0, 2).reshape(-1, 2)
		xforms=xforms.fuse_transpose([0,2,1])
		rays_o = jt.gather(xforms, 0, img_ids)[:, :, 3]
		res = jt.array(self.resolution)
		rays_d = jt.concat([(xy-jt.gather(principal_point, 0, img_ids))
						   * res/focal_length, jt.ones([H*W, 1])], dim=-1)
		rays_d = jt.normalize(jt.gather(xforms, 0, img_ids)[
							  :, :, :3].matmul(rays_d.unsqueeze(2)))
		# resolution W H
		# img H W
		rays_pix = ((xy_int[:, 1]) * H+(xy_int[:, 0])).int()
		# rays origin /dir   rays hit point offset
		return rays_o, rays_d, rays_pix
	
	# use for test
	def generate_rays_with_pose(self, pose, H, W):
		nray = H*W
		pose = self.matrix_nerf2ngp(pose, self.scale, self.offset)
		focal_length = self.focal_lengths[:1].expand(nray, -1)
		xforms = pose.unsqueeze(0).expand(nray, -1, -1)
		principal_point = self.metadata[:1, 4:6].expand(nray, -1)
		xy = jt.stack(jt.meshgrid((jt.linspace(0, H-1, H)+0.5)/H, (jt.linspace(0,
					  W-1, W)+0.5)/W), dim=-1).permute(1, 0, 2).reshape(-1, 2)
		xy_int = jt.stack(jt.meshgrid(jt.linspace(
			0, H-1, H), jt.linspace(0, W-1, W)), dim=-1).permute(1, 0, 2).reshape(-1, 2)
		rays_o = xforms[:, :, 3]
		res = jt.array(self.resolution)
		rays_d = jt.concat([
			(xy-principal_point) * res/focal_length, 
			jt.ones([H*W, 1])
		], dim=-1)
		rays_d = jt.normalize(xforms[:, :, :3].matmul(rays_d.unsqueeze(2)))
		return rays_o, rays_d

	def matrix_nerf2ngp(self, matrix, scale, offset):
		matrix[:, 0] *= self.correct_pose[0]
		matrix[:, 1] *= self.correct_pose[1]
		matrix[:, 2] *= self.correct_pose[2]
		matrix[:, 3] = matrix[:, 3] * scale + offset
		# cycle
		matrix=matrix[[1,2,0]]
		return matrix

	def matrix_ngp2nerf(self, matrix, scale, offset):
		matrix=matrix[[2,0,1]]
		matrix[:, 0] *= self.correct_pose[0]
		matrix[:, 1] *= self.correct_pose[1]
		matrix[:, 2] *= self.correct_pose[2]
		matrix[:, 3] = (matrix[:, 3] - offset) / scale
		return matrix



def rt_inverse(R, T):
	# input: rt pose
	# w2c->c2w or c2w->w2c
	R = R.transpose(1, 0)
	T = -R @ T
	return R, T


def get_pose_from_xml(xml_path):
	pose_xml = parse(xml_path)
	node_0, node_1 = pose_xml.getElementsByTagName("data")
	R = list(map(float, node_0.childNodes[0].data.strip().split()))
	R = np.asarray(R, dtype=np.float32).reshape((3,3))
	
	T = list(map(float, node_1.childNodes[0].data.strip().split()))
	T = np.asarray(T, dtype=np.float32).reshape((3,1))

	# inverse R,T
	R,T = rt_inverse(R, T)
	return np.concatenate((R,T), axis=-1)


def get_intrinsic_from_xml(xml_path):
	intrinsic_xml = parse(xml_path)
	node_0, node_1 = intrinsic_xml.getElementsByTagName("data")
	K = list(map(float, node_0.childNodes[0].data.strip().split()))
	K = np.asarray(K, dtype=np.float32).reshape((3, 3))

	D = list(map(float, node_1.childNodes[0].data.strip().split()))
	D = np.asarray(D, dtype=np.float32).reshape((5, ))
	return K, D


'''

self.image_data.append(img[crop_y: crop_y + crop_H, crop_x : crop_x + crop_W, :].reshape(-1, 3))
# index
i, j = np.meshgrid(np.arange(crop_W, dtype=np.float32), np.arange(crop_H, dtype=np.float32), indexing='xy') # (cH, cW)
i = i + crop_x
j = j + crop_y
ij = np.stack((i,j), axis=2) # ch, cw, 2
# stack i,j
self.image_index.append(ij.reshape(-1, 2)) 
self.croped_info.append([crop_y, crop_x, crop_H, crop_W])

'''
