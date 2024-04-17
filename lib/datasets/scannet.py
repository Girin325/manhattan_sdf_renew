import os
import torch
import numpy as np
from tqdm import tqdm
import cv2
from lib.config import cfg


WALL_SEMANTIC_ID = 80
FLOOR_SEMANTIC_ID = 160


class Dataset(torch.utils.data.Dataset):
    def __init__(self, **kwargs):
        super(Dataset, self).__init__()
        data_root, split, scene = kwargs['data_root'], kwargs['split'], kwargs['scene'] #data/scannet // train // 0050_00
        self.instance_dir = f'{data_root}/{scene}'
        self.split = split
        assert os.path.exists(self.instance_dir)

        image_dir = '{0}/images'.format(self.instance_dir)
        self.image_list = os.listdir(image_dir)
        self.image_list.sort(key=lambda _:int(_.split('.')[0]))
        self.n_images = len(self.image_list)
        
        self.intrinsic_all = []
        self.c2w_all = []
        self.rgb_images = []

        self.semantic_deeplab = []
        self.depth_colmap = []

        intrinsic = np.loadtxt(f'{self.instance_dir}/intrinsic.txt')[:3, :3] #3x3으로 슬라이싱

        for imgname in tqdm(self.image_list, desc='Loading dataset'):
            c2w = np.loadtxt(f'{self.instance_dir}/pose/{imgname[:-4]}.txt')    #이미지의 포즈 정보 저장
            self.c2w_all.append(c2w)
            self.intrinsic_all.append(intrinsic)

            rgb = cv2.imread(f'{self.instance_dir}/images/{imgname[:-4]}.png')  #이미지의 bgr정보 읽어와 rgb로 저장
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            rgb = (rgb.astype(np.float32) / 255).transpose(2, 0, 1)             #이미지 차원 변경 -> 이미지는 (채널, 높이, 너비)로
            _, self.H, self.W = rgb.shape
            rgb = rgb.reshape(3, -1).transpose(1, 0)
            self.rgb_images.append(rgb)

            if self.split == 'train':
                depth_path = f'{self.instance_dir}/depth_colmap/{imgname[:-4]}.npy' #depth_colmap 읽어와 변수 저장
                if os.path.exists(depth_path):
                    depth_colmap = np.load(depth_path)
                    depth_colmap[depth_colmap > 2.0] = 0
                else:
                    depth_colmap = np.zeros((self.H, self.W), np.float32)
                depth_colmap = depth_colmap.reshape(-1)
                self.depth_colmap.append(depth_colmap)
                
                semantic_deeplab = cv2.imread(f'{self.instance_dir}/semantic_deeplab/{imgname[:-4]}.png', -1)
                semantic_deeplab = semantic_deeplab.reshape(-1)
                wall_mask = semantic_deeplab == WALL_SEMANTIC_ID    # WALL_SEMANTIC_ID에 해당하는 픽셀을 true로 하는 이진 마스크 생성
                floor_mask = semantic_deeplab == FLOOR_SEMANTIC_ID
                bg_mask = ~(wall_mask | floor_mask)                 # 벽, 바닥 제외한 영역에 해당하는 픽셀을 true로 하는 이진 마스크 생성
                semantic_deeplab[wall_mask] = 1                     # wall_semantic_id에 해당하는 픽셀 1로
                semantic_deeplab[floor_mask] = 2
                semantic_deeplab[bg_mask] = 0
                self.semantic_deeplab.append(semantic_deeplab)

    def __len__(self):
        return len(self.image_list) #dataset length: 465

    def __getitem__(self, idx):
        c2w, intrinsic = self.c2w_all[idx], self.intrinsic_all[idx]

        ret = {'rgb': self.rgb_images[idx]} #딕셔너리 ret 초기화 후, RGB 이미지를 해당 딕셔너리에 추가함

        if self.split == 'train':
            rays = self.gen_rays(c2w, intrinsic)    #광선 생성
            ret['rays'] = rays
            ret['semantic_deeplab'] = self.semantic_deeplab[idx]
            ret['depth_colmap'] = self.depth_colmap[idx]

            ids = np.random.choice(len(rays), cfg.train.N_rays, replace=False)  #생성된 광선 중 랜덤으로 cfg.train.N_rays 개수만큼 선택
            for k in ret:
                ret[k] = ret[k][ids]
        
        else:
            ret['c2w'] = c2w
            ret['intrinsic'] = intrinsic

        ret.update({'meta': {'h': self.H, 'w': self.W, 'filename': self.image_list[idx]}})
        return ret

    def gen_rays(self, c2w, intrinsic):
        H, W = self.H, self.W
        rays_o = c2w[:3, 3]
        X, Y = np.meshgrid(np.arange(W), np.arange(H))  #이미지 픽셀 좌표 생성 width: 640 / height: 480
        XYZ = np.concatenate((X[:, :, None], Y[:, :, None], np.ones_like(X[:, :, None])), axis=-1)  #이미지 픽셀 좌표를 행렬로 변환 후, 마지막 열에 1 추가해 확장
        XYZ = XYZ @ np.linalg.inv(intrinsic).T          # intrinsic 매트릭스의 역행렬 곱해서 이미지 픽셀 좌표를 카메라 좌표계로 변환
        XYZ = XYZ @ c2w[:3, :3].T                       # 카메라 좌표계에서 월드 좌표계로 변환
        rays_d = XYZ.reshape(-1, 3)                     # 변환된 자표를 1차원 배열로 평탄화 해서 저장

        rays = np.concatenate([rays_o[None].repeat(H*W, axis=0), rays_d], axis=-1)  #원점 rays_o를 H*W번 복제하고, 광선의 방향 rays_d와 연결하여 모든 광선을 생성
        return rays.astype(np.float32)
