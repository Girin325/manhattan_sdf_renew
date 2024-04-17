import torch
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    OpenGLPerspectiveCameras,
    PointLights,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
)

# 필요한 라이브러리가 설치되어 있는지 확인합니다.
# 예: pip install torch torchvision pytorch3d

# 오브젝트를 로드합니다 (이 예제에서는 PyTorch3D에서 제공하는 오브젝트를 사용).
mesh = load_objs_as_meshes(["data/trained_model/manhattan_sdf_renew/scannet_0084_00/latest.pth"], device=torch.device("cuda:0"))

# 카메라 설정
R, T = look_at_view_transform(dist=2.7, elev=10, azim=90)
cameras = OpenGLPerspectiveCameras(device=torch.device("cuda:0"), R=R, T=T)

# 조명 설정
lights = PointLights(location=[[0.0, 0.0, -3.0]], device=torch.device("cuda:0"))

# 렌더러 설정
raster_settings = RasterizationSettings(
    image_size=512,
    blur_radius=0.0,
    faces_per_pixel=1,
)
renderer = MeshRenderer(
    rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
    shader=SoftPhongShader(device=torch.device("cuda:0"), cameras=cameras, lights=lights)
)

# 렌더링 실행
images = renderer(mesh)

# 결과 이미지 출력
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 10))
plt.imshow(images[0, ..., :3].cpu().numpy())
plt.grid(False)
plt.axis('off')
