from .base import SDFNet, RadianceNet, SemanticNet
from .ray_sampler import sdf_to_sigma, fine_sample

import copy
import functools
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.utils.net_utils import batchify_query
from lib.config import cfg
import pdb

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.speed_factor = cfg.model.speed_factor                                          # 학습률로 예상함
        ln_beta_init = np.log(cfg.model.beta_init) / self.speed_factor                      # 여기서 beta는 학습 가능한 파라미터임
        self.ln_beta = nn.Parameter(data=torch.Tensor([ln_beta_init]), requires_grad=True)  # 위에 걸 파이토치 텐서로 변환해서 data에 저장, 연산 추적

        self.sdf_net = SDFNet()
        self.radiance_net = RadianceNet()
        self.semantic_net = SemanticNet()
        #self.semantic_net = UNet()

    def forward_ab(self):
        beta = torch.exp(self.ln_beta * self.speed_factor)
        return 1./beta, beta

    def forward_surface(self, x: torch.Tensor):
        sdf = self.sdf_net.forward(x)
        return sdf        

    def forward_surface_with_nablas(self, x: torch.Tensor):
        sdf, nablas, h = self.sdf_net.forward_with_nablas(x)
        return sdf, nablas, h

    def forward(self, x:torch. Tensor, view_dirs: torch.Tensor):    #raduabce_net?
        sdf, nablas, geometry_feature = self.forward_surface_with_nablas(x)
        radiances = self.radiance_net.forward(x, view_dirs, nablas, geometry_feature)
        semantics = self.semantic_net.forward(x, geometry_feature)
        return radiances, semantics, sdf, nablas
    
    def forward_semantic(self, x:torch. Tensor):
        sdf, nablas, geometry_feature = self.forward_surface_with_nablas(x)
        semantics = self.semantic_net.forward(x, geometry_feature)
        return semantics


def volume_render(
    rays_o, 
    rays_d,
    model: MLP,
    near=0.0,
    far=2.0,
    perturb = True,
    ):

    device = rays_o.device
    rayschunk = cfg.sample.rayschunk    #한 번에 처리할 ray의 개수
    netchunk = cfg.sample.netchunk      #신경망이 한 번에 처리하는 데이터의 크기
    N_samples = cfg.sample.N_samples
    N_importance = cfg.sample.N_importance
    max_upsample_steps = cfg.sample.max_upsample_steps
    max_bisection_steps = cfg.sample.max_bisection_steps
    epsilon = cfg.sample.epsilon

    DIM_BATCHIFY = 1    # 배치 처리 / 차원을 나타내는 값
    B = rays_d.shape[0]  # batch_size
    flat_vec_shape = [B, -1, 3]     # -1: 차원 자동 조정

    rays_o = torch.reshape(rays_o, flat_vec_shape).float()
    rays_d = torch.reshape(rays_d, flat_vec_shape).float()

    depth_ratio = rays_d.norm(dim=-1)
    rays_d = F.normalize(rays_d, dim=-1)
    
    batchify_query_ = functools.partial(batchify_query, chunk=netchunk, dim_batchify=DIM_BATCHIFY)

    def render_rayschunk(rays_o: torch.Tensor, rays_d: torch.Tensor):

        view_dirs = rays_d  #ray의 방향
        
        prefix_batch = [B]  #B: 배치의 크기
        N_rays = rays_o.shape[-2] #뒤에서 2번째 차원의 크기
        #
        # if cfg.train.epoch >= 0 and cfg.train.epoch < 1250:
        #     N_rays = 512
        # elif cfg.train.epoch >= 1250 and cfg.train.epoch < 2500:
        #     N_rays = 256
        # elif cfg.train.epoch >= 2500 and cfg.train.epoch < 3750:
        #     N_rays = 128
        # else:
        #     N_rays = 64
        # print(N_rays)
        nears = near * torch.ones([*prefix_batch, N_rays, 1]).to(device)    #near, far 값을 레이 수만큼 반복해 나타낸 텐서
        fars = far * torch.ones([*prefix_batch, N_rays, 1]).to(device)

        _t = torch.linspace(0, 1, N_samples).float().to(device) #0부터 1까지 균등하게 간격 두고 N_samples 수만큼 샘플링 시킴
        d_coarse = nears * (1 - _t) + fars * _t                 #초기 ray 거리를 균등하게 샘플링
        alpha, beta = model.forward_ab()
        with torch.no_grad():                                   # 더 많은 샘플링 위해 4배의 _t 생성, d_init 계산
            _t = torch.linspace(0, 1, N_samples*4).float().to(device)
            d_init = nears * (1 - _t) + fars * _t
            
            d_fine, beta_map, iter_usage = fine_sample(         # fine_sample 함수
                model.forward_surface, d_init, rays_o, rays_d, 
                alpha_net=alpha, beta_net=beta, far=fars, 
                eps=epsilon, max_iter=max_upsample_steps, max_bisection=max_bisection_steps, 
                final_N_importance=N_importance, perturb=perturb, 
                N_up=N_samples*4
            )

        d_all = torch.cat([d_coarse, d_fine], dim=-1)   #마지막 차원 / 전체 레이의 거리
        d_all, _ = torch.sort(d_all, dim=-1)            # 정렬
        pts = rays_o[..., None, :] + rays_d[..., None, :] * d_all[..., :, None] #각 레이에 대해 거리에 따른 3d 포인트 계산
        
        radiances, semantics, sdf, nablas = batchify_query_(model.forward, pts, view_dirs.unsqueeze(-2).expand_as(pts)) # 포인트에 대해 계산 후 저장
        sigma = sdf_to_sigma(sdf, alpha, beta)          # sdf 값 이용해 시그마 계산
            
        delta_i = d_all[..., 1:] - d_all[..., :-1]      # 거리 간 차이 계산
        p_i = torch.exp(-F.relu_(sigma[..., :-1] * delta_i))    # 광속 계산

        tau_i = (1 - p_i + 1e-10) * (           # 누적된 투과도 계산
            torch.cumprod(
                torch.cat(
                    [torch.ones([*p_i.shape[:-1], 1], device=device), p_i], dim=-1), 
                dim=-1)[..., :-1]
            )

        rgb_map = torch.sum(tau_i[..., None] * radiances[..., :-1, :], dim=-2)      #누적된 투과도 이용해 rgb, semantic 맵 계산
        semantic_map = torch.sum(tau_i[..., None] * semantics[..., :-1, :], dim=-2)
        
        distance_map = torch.sum(tau_i / (tau_i.sum(-1, keepdim=True)+1e-10) * d_all[..., :-1], dim=-1)
        depth_map = distance_map / depth_ratio
        acc_map = torch.sum(tau_i, -1)

        ret_i = OrderedDict([       #결과 저장
            ('rgb', rgb_map),
            ('semantic', semantic_map),
            ('distance', distance_map),
            ('depth', depth_map),
            ('mask_volume', acc_map)
        ])
        # print(ret_i)
        surface_points = rays_o + rays_d * distance_map[..., None]  #distance map 이용해 표면의 점 계산
        _, surface_normals, _ = model.sdf_net.forward_with_nablas(surface_points.detach())  # 표면 점에서 SDF, nablas 값 얻음
        ret_i['surface_normals'] = surface_normals      #surface_normals: 표면의 법선 벡터

        # normals_map = F.normalize(nablas, dim=-1)
        # N_pts = min(tau_i.shape[-1], normals_map.shape[-2])
        # normals_map = (normals_map[..., :N_pts, :] * tau_i[..., :N_pts, None]).sum(dim=-2)
        # ret_i['normals_volume'] = normals_map

        ret_i['sdf'] = sdf
        ret_i['nablas'] = nablas
        ret_i['radiance'] = radiances
        ret_i['alpha'] = 1.0 - p_i
        ret_i['p_i'] = p_i
        ret_i['visibility_weights'] = tau_i
        ret_i['d_vals'] = d_all
        ret_i['sigma'] = sigma
        ret_i['beta_map'] = beta_map
        ret_i['iter_usage'] = iter_usage

        return ret_i
        
    ret = {}
    for i in range(0, rays_o.shape[DIM_BATCHIFY], rayschunk):
        ret_i = render_rayschunk(rays_o[:, i:i+rayschunk], rays_d[:, i:i+rayschunk])    # 레이 트레이싱 수행, 결과 ret에 저장
        for k, v in ret_i.items():
            if k not in ret:
                ret[k] = []
            ret[k].append(v)
    for k, v in ret.items():
        ret[k] = torch.cat(v, DIM_BATCHIFY)
    
    alpha, beta = model.forward_ab()        #ret 결과 합쳐서 최종 결과 생성
    alpha, beta = alpha.data, beta.data
    ret['scalars'] = {'alpha': alpha, 'beta': beta}

    return ret


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.model = MLP()  #MLP 모델 초기화
        
        self.theta = nn.Parameter(torch.Tensor([0.]), requires_grad=True)   #각도 초기화
        # <cos(theta), sin(tehta), 0> is $\mathbf{n}_w$ in equation (9)
    
    def forward(self, batch):
        rays = batch['rays']
        #pdb.set_trace()
        rays_o, rays_d = rays[:, :, :3], rays[:, :, 3:6]
        rays_d[rays_d.abs() < 1e-6] = 1e-6

        if self.training:
            near = cfg.train_dataset.near
            far = cfg.train_dataset.far
            pertube = True
        else:
            near = cfg.test_dataset.near
            far = cfg.test_dataset.far
            pertube = False

        return volume_render(
            rays_o,
            rays_d,
            self.model,
            near = near,
            far=far,
            perturb=pertube
        )

