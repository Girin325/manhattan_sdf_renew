import pdb

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd

from lib.config import cfg


class Embedder(nn.Module):      #데이터를 고정된 크기의 벡터로 변환해줌      #아예 첫단 gamma(x) 부분
    def __init__(self, input_dim, max_freq_log2, N_freqs,
                 log_sampling=True, include_input=True,
                 periodic_fns=(torch.sin, torch.cos)):
        '''
        :param input_dim: dimension of input to be embedded
        :param max_freq_log2: log2 of max freq; min freq is 1 by default
        :param N_freqs: number of frequency bands
        :param log_sampling: if True, frequency bands are linerly sampled in log-space
        :param include_input: if True, raw input is included in the embedding
        :param periodic_fns: periodic functions used to embed input
        '''
        super(Embedder, self).__init__()

        self.input_dim = input_dim
        self.include_input = include_input
        self.periodic_fns = periodic_fns

        self.out_dim = 0
        if self.include_input:
            self.out_dim += self.input_dim
            # print(self.input_dim)                                                 3
            # print("1:", self.out_dim)                                             3
        self.out_dim += self.input_dim * N_freqs * len(self.periodic_fns)
        # print("2", self.out_dim)                                                  39
        if log_sampling:
            self.freq_bands = 2. ** torch.linspace(0., max_freq_log2, N_freqs)
        else:
            self.freq_bands = torch.linspace(
                2. ** 0., 2. ** max_freq_log2, N_freqs)

        self.freq_bands = self.freq_bands.numpy().tolist()

    def forward(self, input: torch.Tensor):
        '''
        :param input: tensor of shape [..., self.input_dim]
        :return: tensor of shape [..., self.out_dim]
        '''
        assert (input.shape[-1] == self.input_dim)

        out = []
        if self.include_input:
            out.append(input)

        for i in range(len(self.freq_bands)):
            freq = self.freq_bands[i]
            for p_fn in self.periodic_fns:
                out.append(p_fn(input * freq))
        out = torch.cat(out, dim=-1)

        assert (out.shape[-1] == self.out_dim)
        return out


def get_embedder(multires, input_dim=3): #multiresolution
    if multires < 0:
        return nn.Identity(), input_dim #입력과 동일한 tensor 출력으로 내보냄

    embed_kwargs = {
        "include_input": True,  # needs to be True for ray_bending to work properly
        "input_dim": input_dim,
        "max_freq_log2": multires - 1,
        "N_freqs": multires,
        "log_sampling": True,
        "periodic_fns": [torch.sin, torch.cos],
    }
    # print(multires)                                               6
    embedder_obj = Embedder(**embed_kwargs)
    # print(embedder_obj.out_dim)
    return embedder_obj, embedder_obj.out_dim                       #Embedder(), 39


class DenseLayer(nn.Linear):
    def __init__(self, input_dim: int, out_dim: int, *args, activation=None, **kwargs):
        super().__init__(input_dim, out_dim, *args, **kwargs)
        if activation is None:
            self.activation = nn.ReLU(inplace=True)
        else:
            self.activation = activation

    def forward(self, x):
        out = super().forward(x)
        out = self.activation(out)
        return out


class SDFNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.D = cfg.model.sdf.net_depth    #8
        self.W = cfg.model.net_width        #256
        self.W_geo_feat = cfg.model.feature_width       #256
        self.skips = cfg.model.sdf.skips    #4

        embed_multires = cfg.model.sdf.fr_pos
        self.embed_fn, input_ch = get_embedder(embed_multires)
        # print(embed_multires)

        surface_fc_layers = []
        # NOTE: as in IDR/NeuS, the network's has D+1 layers
        for l in range(self.D+1):
            # decide out_dim
            if l == self.D:
                if self.W_geo_feat > 0:
                    out_dim = 1 + self.W_geo_feat
                else:
                    out_dim = 1
            elif (l+1) in self.skips:
                out_dim = self.W - input_ch  # recude(reduce ??) output dim before the skips layers, as in IDR / NeuS
            else:
                out_dim = self.W
            #print("out_dim:", out_dim)
            # decide in_dim
            if l == 0:
                in_dim = input_ch
            else:
                in_dim = self.W

            if l != self.D:
                layer = DenseLayer(in_dim, out_dim, activation=nn.Softplus(beta=100)) #softplus: sigmoid 적분한 함수/ 양수의 출력값 가짐, 렐루가 0이되는 순간 완화함
            else:
                layer = nn.Linear(in_dim, out_dim)
            #print(l)
            # if true preform preform geometric initialization
            if cfg.model.sdf.geometric_init:        # 레이어 초기화
                #--------------
                # sphere init, as in SAL / IDR.
                #--------------
                if l == self.D: #정규 분포에서 샘플링한 가중치와 상수 바이어스 사용해 초기화
                    nn.init.normal_(layer.weight, mean=np.sqrt(np.pi) / np.sqrt(in_dim), std=0.0001)    #평균 / 표준편차
                    nn.init.constant_(layer.bias, -cfg.model.sdf.radius_init) 
                elif embed_multires > 0 and l == 0:
                    torch.nn.init.constant_(layer.bias, 0.0)
                    torch.nn.init.constant_(layer.weight[:, 3:], 0.0)   # let the initial weights for octaves to be 0.
                    torch.nn.init.normal_(layer.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif embed_multires > 0 and l in self.skips:
                    torch.nn.init.constant_(layer.bias, 0.0)
                    torch.nn.init.normal_(layer.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(layer.weight[:, -(input_ch - 3):], 0.0) # NOTE: this contrains the concat order to be  [h, x_embed]
                else:
                    nn.init.constant_(layer.bias, 0.0)
                    nn.init.normal_(layer.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if cfg.model.sdf.weight_norm:       #가중치 정규화 / weight_norm이 True면 아래 weight_norm 사용해 가중치 정규화 적용
                layer = nn.utils.weight_norm(layer)

            surface_fc_layers.append(layer)
            # print(surface_fc_layers)
            """
[
DenseLayer1(in_features=39, out_features=256, bias=True(activation): Softplus(beta=100, threshold=20)), 
DenseLayer2(in_features=256, out_features=256, bias=True(activation): Softplus(beta=100, threshold=20)), 
DenseLayer3(in_features=256, out_features=256, bias=True(activation): Softplus(beta=100, threshold=20)), 
DenseLayer4(in_features=256, out_features=217, bias=True(activation): Softplus(beta=100, threshold=20)), 
DenseLayer5(in_features=256, out_features=256, bias=True(activation): Softplus(beta=100, threshold=20)), 
DenseLayer6(in_features=256, out_features=256, bias=True(activation): Softplus(beta=100, threshold=20)), 
DenseLayer7(in_features=256, out_features=256, bias=True(activation): Softplus(beta=100, threshold=20)), 
DenseLayer8(in_features=256, out_features=256, bias=True(activation): Softplus(beta=100, threshold=20)), 
Linear9(in_features=256, out_features=257, bias=True)
]

            """
        self.surface_fc_layers = nn.ModuleList(surface_fc_layers)

    def forward(self, x: torch.Tensor, return_h = False):   #순전파 정의
        x = self.embed_fn(x) #텐서 들어가있음
        h = x
        for i in range(self.D):
            if i in self.skips:                                 # 스킵 연결을 위해 현재 feature와 이전 feature를 연결
                # NOTE: concat order can not change! there are special operations taken in intialization.
                h = torch.cat([h, x], dim=-1) / np.sqrt(2)      # torch.cat : 주어진 텐서를 차원에 맞춰 합쳐줌
            h = self.surface_fc_layers[i](h)
        
        out = self.surface_fc_layers[-1](h) #최종 출력 처리
        
        if self.W_geo_feat > 0:
            h = out[..., 1:]
            out = out[..., :1].squeeze(-1)
        else:
            out = out.squeeze(-1)
        
        out = -out  # make it suitable to inside-out scene

        if return_h:
            return out, h
        else:
            return out
    
    def forward_with_nablas(self,  x: torch.Tensor, has_grad_bypass: bool = None):
        has_grad = torch.is_grad_enabled() if has_grad_bypass is None else has_grad_bypass #그래디언트가 활성화되어 있는지 확인, 필요한 경우 그라디언트 활성화
        # force enabling grad for normal calculation
        with torch.enable_grad():
            x = x.requires_grad_(True)
            sdf, h = self.forward(x, return_h=True)
            nabla = autograd.grad(  #그라디언트 계산
                sdf,
                x,
                torch.ones_like(sdf, device=x.device),
                create_graph=has_grad,
                retain_graph=has_grad,
                only_inputs=True
            )[0]
        if not has_grad:    #그라디언트 비활성화 된 경우, 텐서 분리해 반환
            sdf = sdf.detach()
            nabla = nabla.detach()
            h = h.detach()
        return sdf, nabla, h


class RadianceNet(nn.Module):
    def __init__(self):
        super().__init__()

        input_ch_pts = 3 #입력 차원 설정
        input_ch_views = 3

        #radianceNet 설정값들 가져오기
        self.skips = cfg.model.radiance.skips
        self.D = cfg.model.radiance.net_depth
        self.W = cfg.model.net_width
        #특정 임베딩 함수 가져오기
        embed_multires = cfg.model.radiance.fr_pos
        embed_multires_view = cfg.model.radiance.fr_view
        self.embed_fn, input_ch_pts = get_embedder(embed_multires)
        self.embed_fn_view, input_ch_views = get_embedder(embed_multires_view)
        #추가적인 특징 차원 설정
        self.W_geo_feat = cfg.model.feature_width
        in_dim_0 = input_ch_pts + input_ch_views + 3 + self.W_geo_feat
        
        fc_layers = []
        # NOTE: as in IDR/NeuS, the network's has D+1 layers
        for l in range(self.D + 1):
            # decicde out_dim
            if l == self.D:
                out_dim = 3
            else:
                out_dim = self.W
            
            # decide in_dim
            if l == 0:
                in_dim = in_dim_0
            elif l in self.skips:
                in_dim = in_dim_0 + self.W
            else:
                in_dim = self.W
            
            if l != self.D:
                layer = DenseLayer(in_dim, out_dim, activation=nn.ReLU(inplace=True))
            else:
                layer = DenseLayer(in_dim, out_dim, activation=nn.Sigmoid())

            if cfg.model.radiance.weight_norm:
                layer = nn.utils.weight_norm(layer)

            fc_layers.append(layer)

        self.layers = nn.ModuleList(fc_layers)
    
    def forward(
        self, 
        x: torch.Tensor, 
        view_dirs: torch.Tensor, 
        normals: torch.Tensor, 
        geometry_feature: torch.Tensor
    ):
        # calculate radiance field
        x = self.embed_fn(x)
        view_dirs = self.embed_fn_view(view_dirs)
        radiance_input = torch.cat([x, view_dirs, normals, geometry_feature], dim=-1)
        
        h = radiance_input
        for i in range(self.D+1):
            if i in self.skips:
                h = torch.cat([h, radiance_input], dim=-1)
            h = self.layers[i](h)
        return h


class SemanticNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        input_ch_pts = 3
        self.skips = cfg.model.semantic.skips
        self.D = cfg.model.semantic.net_depth
        self.W = cfg.model.net_width
        embed_multires = cfg.model.semantic.fr_pos
        self.embed_fn, input_ch_pts = get_embedder(embed_multires)
        self.W_geo_feat = cfg.model.feature_width
        in_dim_0 = input_ch_pts + self.W_geo_feat
        # print(in_dim_0)                                           259
        fc_layers = []
        # NOTE: as in IDR/NeuS, the network's has D+1 layers
        for l in range(self.D + 1):
            # decicde out_dim
            if l == self.D:
                out_dim = 3
                # print("2: ", out_dim)
            else:
                out_dim = self.W
                # print("1: ", out_dim)
            
            # decide in_dim
            if l == 0:
                in_dim = in_dim_0
                # print("3:", in_dim) 259
            elif l in self.skips:
                in_dim = in_dim_0 + self.W
                # print("2:", in_dim)
            else:
                in_dim = self.W
                # print("1:", in_dim) 256
            
            if l != self.D:
                layer = DenseLayer(in_dim, out_dim, activation=nn.ReLU(inplace=True))
            else:
                layer = DenseLayer(in_dim, out_dim, activation=nn.Sigmoid())
            
            if cfg.model.semantic.weight_norm:
                layer = nn.utils.weight_norm(layer)

            fc_layers.append(layer)
        #print(fc_layers)
        self.layers = nn.ModuleList(fc_layers)
    
    def forward(
        self, 
        x: torch.Tensor, 
        geometry_feature: torch.Tensor):
        # calculate semantic field
        x = self.embed_fn(x)
        semantic_input = torch.cat([x, geometry_feature], dim=-1)
        # print("semantic:input: ", semantic_input)
        # print("shape: ", semantic_input.shape)
        h = semantic_input
        for i in range(self.D+1):
            if i in self.skips:
                h = torch.cat([h, semantic_input], dim=-1)
            h = self.layers[i](h)
        # print("h:", h)
        return h

"""
################################################################

class ConvBlock(nn.Module):
    def __init__(self, input_ch, output_ch):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_ch, output_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(output_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_ch, output_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(output_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UpConv(nn.Module):
    def __init__(self, input_ch, output_ch):
        super(UpConv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(input_ch, output_ch, kernel_size=1),
            nn.BatchNorm2d(output_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.up(x)


class DownConv(nn.Module):
    def __init__(self, input_ch, output_ch):
        super(DownConv, self).__init__()
        self.down = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(input_ch, output_ch)
        )

    def forward(self, x):
        return self.down(x)


class UNet3Plus(nn.Module):
    def __init__(self):
        super(UNet3Plus, self).__init__()
        input_ch_pts = 3
        # Initial convolution block
        self.D = cfg.model.semantic.net_depth
        self.W = cfg.model.net_width
        self.W_geo_feat = cfg.model.feature_width
        self.skips = cfg.model.semantic.skips
        embed_multires = cfg.model.semantic.fr_pos

        self.embed_fn, input_ch_pts = get_embedder(embed_multires)
        in_dim_0 = input_ch_pts + self.W_geo_feat

        fc_layers = []
        self.init_conv = DownConv(in_dim_0, 64)
        fc_layers.append(self.init_conv)
        self.down1 = DownConv(64, 128)
        fc_layers.append(self.down1)
        self.down2 = DownConv(128, 256)
        fc_layers.append(self.down2)
        self.down3 = DownConv(256, 512)
        fc_layers.append(self.down3)
        self.up1_0 = UpConv(512, 256)
        fc_layers.append(self.up1_0)
        self.up2_0 = UpConv(256, 128)
        fc_layers.append(self.up2_0)
        self.up3_0 = UpConv(128, 64)
        fc_layers.append(self.up3_0)
        self.up4_0 = UpConv(64, 32)
        fc_layers.append(self.up4_0)

        self.conv0_1 = ConvBlock(512 + 256, 256)
        fc_layers.append(self.conv0_1)
        self.conv0_2 = ConvBlock(256 + 128, 128)
        fc_layers.append(self.conv0_2)
        self.conv0_3 = ConvBlock(128 + 64, 64)
        fc_layers.append(self.conv0_3)
        self.conv0_4 = ConvBlock(64 + 32, 32)
        fc_layers.append(self.conv0_4)

        self.final_conv = nn.Conv2d(32, 3, kernel_size=1)
        print(fc_layers)
        if cfg.model.semantic.weight_norm:
            self.apply_weight_norm()

        self.layers = nn.ModuleList(fc_layers)

    def apply_weight_norm(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.ConvTranspose2d):
                nn.utils.weight_norm(layer)

    def forward(self, x: torch.Tensor, geometry_feature: torch.Tensor):
        # Initial features
        x0_0 = self.init_conv(x)

        # Downsampled pathways
        x1_0 = self.down1(x0_0)
        x2_0 = self.down2(x1_0)
        x3_0 = self.down3(x2_0)

        semantic_input = self.embed_fn(x3_0)
        semantic_input = torch.cat([semantic_input, geometry_feature], dim=-1)

        x0_1 = self.up1_0(x3_0)
        x0_1 = self.conv0_1(torch.cat([x0_1, x3_0], dim=-1))  # Note the ordering of concatenation

        x0_2 = self.up2_0(x0_1)
        x0_2 = self.conv0_2(torch.cat([x0_2, x2_0], dim=-1))  # Again, pay attention to the ordering

        x0_3 = self.up3_0(x0_2)
        x0_3 = self.conv0_3(torch.cat([x0_3, x1_0], dim=-1))

        x0_4 = self.up4_0(x0_3)
        x0_4 = self.conv0_4(torch.cat([x0_4, x0_0], dim=-1))  # Final concatenation before the output layer

        # Final output
        out = self.final_conv(x0_4)

        return out
"""
"""

class UNet3Plus(nn.Module):
    def __init__(self):
        super(UNet3Plus, self).__init__()
        # Configuration parameters loaded from cfg object, assuming it's defined elsewhere
        self.D = cfg.model.semantic.net_depth
        self.W = cfg.model.net_width
        self.W_geo_feat = cfg.model.feature_width
        self.skips = cfg.model.semantic.skips
        embed_multires = cfg.model.semantic.fr_pos
        
        # Embedding function for input features
        self.embed_fn, input_ch_pts = get_embedder(embed_multires)
        in_dim_0 = input_ch_pts + self.W_geo_feat
        
        # Define downsample and upsample layers
        self.init_conv = DownConv(input_ch_pts, 64)
        self.down1 = DownConv(64, 128)
        self.down2 = DownConv(128, 256)
        self.up1_0 = UpConv(256, 128)
        self.up2_0 = UpConv(128, 64)
        self.up3_0 = UpConv(64, 32)
        
        # Define intermediate convolutions
        self.conv0_1 = ConvBlock(64 + 128, 64)
        self.conv0_2 = ConvBlock(64 + 128 + 64, 64)
        self.conv0_3 = ConvBlock(64 + 128 + 64 + 32, 64)
        
        # Define final output layer
        self.final_conv = nn.Conv2d(64, 3, kernel_size=1)
        
        # Assuming weight normalization is a boolean flag indicating whether to apply weight norm or not
        if cfg.model.semantic.weight_norm:
            self.apply_weight_norm()

    def apply_weight_norm(self):
        # Apply weight normalization to all the conv layers
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.ConvTranspose2d):
                nn.utils.weight_norm(layer)

    def forward(self, x):
        # Initial features
        x0_0 = self.init_conv(x)
        
        # Downsampled pathways
        x1_0 = self.down1(x0_0)
        x2_0 = self.down2(x1_0)
        
        # Upsampled and concatenated features
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up1_0(x1_0)], dim=1))
        x0_2 = self.conv0_2(torch.cat([x0_0, self.up1_0(x1_0), self.up2_0(x2_0)], dim=1))
        x0_3 = self.conv0_3(torch.cat([x0_0, self.up
"""

"""
        in_features=259, out_features=256, bias=True
  (activation): ReLU(inplace=True)
), DenseLayer(
  in_features=256, out_features=256, bias=True
  (activation): ReLU(inplace=True)
), DenseLayer(
  in_features=256, out_features=256, bias=True
  (activation): ReLU(inplace=True)
), DenseLayer(
  in_features=256, out_features=256, bias=True
  (activation): ReLU(inplace=True)
), DenseLayer(
  in_features=256, out_features=3, bias=True
  (activation): Sigmoid()
)]

"""
"""
        self.init_conv = ConvBlock(in_channels, 259)
        self.down1 = DownConv(259, 512)
        self.down2 = DownConv(512, 1024)

        self.up2_0 = UpConv(1024, 64)
        self.up1_0 = UpConv(512, 64)
        # self.up1_0 = UpConv(128, 3)

        self.conv0_2 = ConvBlock(259, 64)
        self.conv0_3 = ConvBlock(512, 64)

        self.final_conv = nn.Conv2d(64, out_dim)

        fc_layers.append(layer)
"""
"""
        # Downsample

        self.down1 = DownConv(259, 512)
        self.down2 = DownConv(512, 1024)

        # Upsample
        self.up3_0 = UpConv(512, 64)
        self.up2_0 = UpConv(256, 64)
        self.up1_0 = UpConv(128, 64)

        # Intermediate convolutions
        self.conv0_1 = ConvBlock(128, 64)
        self.conv0_2 = ConvBlock(192, 64)
        self.conv0_3 = ConvBlock(256, 64)

        # Final output layer
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
"""

