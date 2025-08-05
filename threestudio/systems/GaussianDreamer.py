import math
from dataclasses import dataclass, field
import torch
import threestudio
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from threestudio.systems.base import BaseLift3DSystem
from threestudio.utils.ops import binary_cross_entropy, dot
from threestudio.utils.typing import *
from gaussiansplatting.gaussian_renderer import render
from gaussiansplatting.scene import Scene, GaussianModel
from gaussiansplatting.arguments import ModelParams, PipelineParams, get_combined_args,OptimizationParams
from gaussiansplatting.scene.cameras import Camera, MiniCam
from argparse import ArgumentParser, Namespace
import os
import random
from pathlib import Path
from plyfile import PlyData, PlyElement
from gaussiansplatting.utils.sh_utils import SH2RGB
from gaussiansplatting.scene.gaussian_model import BasicPointCloud
import numpy as np
import torchvision
import torchvision.transforms.functional as TFunc
import time
import cv2
from torchvision.utils import make_grid, save_image
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

# from transformers import SamProcessor, SamModel
# import torch.nn.functional as F

# from shap_e.diffusion.sample import sample_latents
# from shap_e.diffusion.gaussian_diffusion import diffusion_from_config as diffusion_from_config_shape
# from shap_e.models.download import load_model, load_config
# from shap_e.util.notebooks import create_pan_cameras, decode_latent_images, gif_widget
# from shap_e.util.notebooks import decode_latent_mesh
import io  
from PIL import Image  
# import open3d as o3d

from threestudio.utils.poser import Skeleton
import torch.nn.functional as F

def load_ply(path,save_path):
    C0 = 0.28209479177387814
    def SH2RGB(sh):
        return sh * C0 + 0.5
    plydata = PlyData.read(path)

    xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                    np.asarray(plydata.elements[0]["y"]),
                    np.asarray(plydata.elements[0]["z"])),  axis=1)

    features_dc = np.zeros((xyz.shape[0], 3, 1))
    features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
    features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])
    color = SH2RGB(features_dc[:,:,0])

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(xyz)
    point_cloud.colors = o3d.utility.Vector3dVector(color)
    o3d.io.write_point_cloud(save_path, point_cloud)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def axis_angle_to_matrix(axis_angle):
    angle = torch.norm(axis_angle, dim=1, keepdim=True)
    axis = axis_angle / (angle + 1e-8)
    x, y, z = axis[:, 0], axis[:, 1], axis[:, 2]

    cos = torch.cos(angle).squeeze(1)
    sin = torch.sin(angle).squeeze(1)
    one_minus_cos = 1 - cos

    rot = torch.zeros((axis.shape[0], 3, 3), device=axis.device)
    rot[:, 0, 0] = cos + x * x * one_minus_cos
    rot[:, 0, 1] = x * y * one_minus_cos - z * sin
    rot[:, 0, 2] = x * z * one_minus_cos + y * sin

    rot[:, 1, 0] = y * x * one_minus_cos + z * sin
    rot[:, 1, 1] = cos + y * y * one_minus_cos
    rot[:, 1, 2] = y * z * one_minus_cos - x * sin

    rot[:, 2, 0] = z * x * one_minus_cos - y * sin
    rot[:, 2, 1] = z * y * one_minus_cos + x * sin
    rot[:, 2, 2] = cos + z * z * one_minus_cos

    return rot

def draw_camera_frustum(ax, c2w: torch.Tensor, intrinsics: torch.Tensor, image_size: tuple, scale=0.1, color='r'):
    """
    Draws a camera frustum in 3D.
    c2w: (4, 4) camera-to-world matrix
    intrinsics: (3, 3)
    image_size: (H, W)
    """
    H, W = image_size
    fx = intrinsics[0, 0].item()
    fy = intrinsics[1, 1].item()
    cx = intrinsics[0, 2].item()
    cy = intrinsics[1, 2].item()

    # Create image plane corners in pixel space
    corners = np.array([
        [0, 0],
        [W, 0],
        [W, H],
        [0, H]
    ])

    # Project to normalized image plane (camera space, z = 1)
    cam_corners = []
    for (u, v) in corners:
        x = (u - cx) / fx
        y = (v - cy) / fy
        cam_corners.append([x, y, 1])
    cam_corners = np.array(cam_corners).T  # (3, 4)

    # Scale and transform to world space
    cam_corners *= scale  # make frustum visible
    cam_corners = np.vstack((cam_corners, np.ones((1, 4))))  # (4, 4)
    world_corners = (c2w @ torch.tensor(cam_corners).float().to(c2w.device)).cpu().numpy()[:3]

    cam_origin = c2w[:3, 3].cpu().numpy()

    # Draw lines from origin to corners
    for i in range(4):
        ax.plot(
            [cam_origin[0], world_corners[0, i]],
            [cam_origin[1], world_corners[1, i]],
            [cam_origin[2], world_corners[2, i]],
            color=color
        )

    # Draw image plane edges
    for i in range(4):
        j = (i + 1) % 4
        ax.plot(
            [world_corners[0, i], world_corners[0, j]],
            [world_corners[1, i], world_corners[1, j]],
            [world_corners[2, i], world_corners[2, j]],
            color=color
        )


def debug_render_splats_and_camera(gaussian_model, c2w, intrinsics, image_size, save_path="debug_render.png"):
    xyz = gaussian_model.get_xyz.detach().cpu().numpy()

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], s=1, alpha=0.5, label='Splats')

    draw_camera_frustum(ax, c2w, intrinsics, image_size, scale=0.3, color='r')
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Gaussian Splats and Camera Frustum")
    ax.view_init(elev=20, azim=90)
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved debug render to {save_path}")

def sample_colors(image, verts2d):
    H, W = image.shape[1:]  # CxHxW

    # Normalize verts2d to [-1, 1] for grid_sample
    verts_norm = verts2d.clone()
    verts_norm[:, 0] = (verts_norm[:, 0] / (W - 1)) * 2 - 1
    verts_norm[:, 1] = (verts_norm[:, 1] / (H - 1)) * 2 - 1
    verts_norm = verts_norm.unsqueeze(0).unsqueeze(2)  # [1, N, 1, 2]

    # Image must be float and [1, C, H, W]
    image = image.float().unsqueeze(0) / 255.0

    # Sample colors
    sampled = F.grid_sample(image, verts_norm, align_corners=True, mode='bilinear')  # [1, C, N, 1]
    colors = sampled.squeeze(0).squeeze(2).permute(1, 0)  # [N, C]
    return colors


@threestudio.register("gaussiandreamer-system")
class GaussianDreamer(BaseLift3DSystem):
    @dataclass
    class Config(BaseLift3DSystem.Config):
        radius: float = 4
        texture_structure_joint: bool = False
        controlnet: bool = False
        smplx_path: str = "/path/to/smplx/model"
        pts_num: int = 100000

        disable_hand_densification: bool = False
        hand_radius: float = 0.05
        densify_prune_start_step: int = 300
        densify_prune_end_step: int = 2100
        densify_prune_interval: int = 300
        size_threshold: int = 20
        size_threshold_fix_step: int = 1500
        half_scheduler_max_step: int = 1500
        max_grad: float = 0.0002
        gender: str = 'neutral'
        prune_only_start_step: int = 2400
        prune_only_end_step: int = 3300
        prune_only_interval: int = 300
        prune_size_threshold: float = 0.008
        masking: bool = False
        masking_each_own: bool = False

        apose: bool = True
        bg_white: bool = False
        use_img: bool = False
        img_path: str = ""

        prompt_options_json: Optional[dict] = None

    cfg: Config
    def configure(self) -> None:
        self.radius = self.cfg.radius
        self.gaussian = GaussianModel(sh_degree = 0)
        self.background_tensor = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda") if self.cfg.bg_white else torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")

        self.parser = ArgumentParser(description="Training script parameters")
        self.pipe = PipelineParams(self.parser)

        self.texture_structure_joint = self.cfg.texture_structure_joint
        self.masking = self.cfg.masking
        self.masking_each_own = self.cfg.masking_each_own
        self.controlnet = self.cfg.controlnet

        # old method with /thirdparty/
        self.sam = sam_model_registry['vit_h'](checkpoint='/n/holylfs05/LABS/pfister_lab/Lab/coxfs01/pfister_lab2/Lab/abenahmedk/sam/sam_vit_h_4b8939.pth')
        self.sam.to("cuda")
        self.mask_generator = SamAutomaticMaskGenerator(self.sam)
        #self.mask_generator = SamPredictor(self.sam)

        # self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
        # self.sam_model = SamModel.from_pretrained("facebook/sam-vit-base").to(self._device)        

        self.scheduled_masks = []
        self.last_mask_update_step = 0
        self.mask_update_interval = 10

        self.use_img = self.cfg.use_img
        self.img_path = self.cfg.img_path
        self.precolor = None

        if self.use_img:
            model = torch.jit.load('nlf/models/nlf_l_multi.torchscript').cuda().eval()
            self.image = TFunc.rotate(TFunc.hflip(torchvision.io.read_image(self.img_path)), angle=180).cuda()

            frame_batch = self.image.unsqueeze(0)
            with torch.inference_mode(), torch.device('cuda'):
                pred = model.detect_smpl_batched(frame_batch, model_name='smplx')

            pose = pred['pose'][0]
            transl = pred['trans'][0]
            betas = pred['betas'][0]
            
            import smplx
            bm = smplx.SMPLX(os.path.join(self.cfg.smplx_path, "smplx"), use_pca=False).cuda().eval()
            res = bm(global_orient=pose[:, :3],
                        body_pose=pose[:, 3:22*3],
                        betas=betas,
                        transl=transl,
                        left_hand_pose=pose[:, 25*3:40*3],
                        right_hand_pose=pose[:, 40*3:55*3],
                        jaw_pose=pose[:, 22*3:23*3],
                        leye_pose=pose[:, 23*3:24*3],
                        reye_pose=pose[:, 24*3:25*3],
                        expression=torch.zeros_like(betas[:, :10])
                    )

           

            def project_vertices(coords3d, intrinsic_matrix):
                projected = coords3d / torch.maximum(
                    torch.tensor(0.001), torch.tensor(coords3d[..., 2:]))
                return torch.einsum('bnk,bjk->bnj', projected, intrinsic_matrix[..., :2, :])

            self.v3d = pred['vertices3d'][0][0].detach().cpu().numpy()
            faces = bm.faces

            import trimesh

            mesh = trimesh.Trimesh(self.v3d, faces, process=False)
            sampled_points, _ = trimesh.sample.sample_surface(mesh, self.cfg.pts_num)

            self.sampled_points_tensor = torch.tensor(sampled_points, dtype=torch.float32, device='cuda')

            from nlf.pt import ptu3d
            self.intrinsic = ptu3d.intrinsic_matrix_from_field_of_view(55, self.image.shape[1:3])

            verts2d = project_vertices(self.sampled_points_tensor.unsqueeze(0), self.intrinsic.to(self.sampled_points_tensor.device)).squeeze(0)
            self.precolor = sample_colors(self.image, verts2d)
          

            R_body_to_world = axis_angle_to_matrix(pose[:, :3])

            t = transl[0]
            self.extrinsic = torch.eye(4, dtype=torch.float32).cuda()
            self.extrinsic[:3, :3] = R_body_to_world[0]
            self.extrinsic[:3, 3] = t
            self.extrinsic = torch.inverse(self.extrinsic)


            if self.texture_structure_joint:
                self.skel = Skeleton(humansd_style=True, apose=self.cfg.apose)
                t = self.skel.load_smplx(self.cfg.smplx_path, pred=pred, gender=self.cfg.gender)
                #t = self.skel.scale(-10)

            else:
                self.skel = Skeleton(apose=self.cfg.apose)
                self.skel.load_smplx(self.cfg.smplx_path, pred=pred, gender=self.cfg.gender)
                #t = self.skel.scale(-10) 

            

        else:
            if self.texture_structure_joint:
                # skel
                self.skel = Skeleton(humansd_style=True, apose=self.cfg.apose)
                # self.skel.load_json('17point')
                self.skel.load_smplx(self.cfg.smplx_path, gender=self.cfg.gender)
                #self.skel.scale(-10)
            else:
                # skel
                self.skel = Skeleton(apose=self.cfg.apose)
                # self.skel.load_json('8head')
                self.skel.load_smplx(self.cfg.smplx_path, gender=self.cfg.gender)
                #self.skel.scale(-10)

            
        self.skel.normalize()
        self.skel.scale(-10)
        dev = self.sampled_points_tensor.device

        ntensor = self.sampled_points_tensor.clone().detach().cpu().numpy()
        tmin = ntensor.min(0)
        tmax = ntensor.max(0)
        c = (tmin + tmax) / 2
        s = 0.6 / (np.max(tmax - tmin))
        ctensor = torch.tensor(c, device=dev)
        stensor = torch.tensor(s, device=dev)

        self.sampled_points_tensor = (self.sampled_points_tensor - ctensor) * stensor
        self.sampled_points_tensor[:, [1, 2]] = self.sampled_points_tensor[:, [2, 1]]
        self.sampled_points_tensor *= (1.1 ** 10)
            
        
        self.cameras_extent = 4.0
        
        self.mobilenet = torchvision.models.mobilenet_v2(pretrained=True).eval()
        self.feature_extractor = self.mobilenet.features

        import json
        with open("smplx_vert_segmentation.json", "r") as j:
            self.vertexmap = json.load(j)
        
        import cubvh

        self.BVH = cubvh.cuBVH(self.skel.vertices, self.skel.faces)

        self.segs = {
            "full": ["head", "leftEye", "rightEye", "eyeballs", "neck", "leftShoulder", "rightShoulder", "spine", "spine1", "spine2", "leftArm", "leftForeArm", "rightArm", "rightForeArm", "leftHand", "leftHandIndex1", "rightHand", "rightHandIndex1", "hips", "leftLeg", "leftUpLeg", "rightLeg", "rightUpLeg", "leftToeBase", "leftFoot", "rightToeBase", "rightFoot"],
            "head": ["head", "leftEye", "rightEye", "eyeballs", "neck"],
            "torso": ["leftShoulder", "rightShoulder", "spine", "spine1", "spine2"],
            "left_arm": ["leftArm", "leftForeArm"],
            "right_arm": ["rightArm", "rightForeArm"],
            "left_hand": ["leftHand", "leftHandIndex1"],
            "right_hand": ["rightHand", "rightHandIndex1"],
            "waist": ["hips"],
            "left_leg": ["leftLeg", "leftUpLeg"],
            "right_leg": ["rightLeg", "rightUpLeg"],
            "left_foot": ["leftToeBase", "leftFoot"],
            "right_foot": ["rightToeBase", "rightFoot"],
        }

        self.seg_order = [
                    "head", "torso", "left_arm", "right_arm",
                    "left_hand", "right_hand", "waist", "left_leg",
                    "right_leg", "left_foot", "right_foot"      
                          ]
        
        self.cycle_len = len(self.seg_order)

    def generate_all_sam_masks_og(self, image_tensor: torch.Tensor):
        """
        image_tensor: [B, H, W, 3] floats in [0,1]
        Produces self.scheduled_masks as a list of 2D masks (H×W), filtered by size.
        """
        start_time = time.time()
        B, H, W, C = image_tensor.shape
        all_masks = []
        min_area = H * W * 0.005  # drop masks smaller than 5% of image area
        max_area = H * W * 0.1 
        print(f"{H = }, {W = }")
        print("SAM model on:", next(self.sam.parameters()).device)

        print("Predictor on:", self.mask_generator.predictor.model.device)


        for i in range(B):
            # to H×W×3 uint8
            start = time.time()
            img = (image_tensor[i].detach().cpu().numpy() * 255).astype(np.uint8)
            print(f"moving to cpu takes {time.time() - start}")
            start = time.time()
            # print("########################")
            # print("GPU memory (before):", torch.cuda.memory_allocated())
            print("SAM model on:", next(self.sam.parameters()).device)

            masks_info = self.mask_generator.generate(img)
            # print("GPU memory (after):", torch.cuda.memory_allocated())
            # print("########################")
            print(f"generating this image took {time.time() - start}")

            start = time.time()
            for m in masks_info:
                seg = m["segmentation"]  # H×W bool/uint8
                if seg.sum() < min_area or seg.sum() > max_area:
                    continue
                # store as float32 mask H×W on device
                mask_tensor = torch.from_numpy(seg.astype(np.float32)).to(image_tensor.device)
                print(f"{mask_tensor.shape = }")
                if self.masking_each_own:
                    all_masks.append((mask_tensor, i))
                else:
                    all_masks.append(mask_tensor)
            print(f"The masks checking took {time.time() - start}")

        print(f"SAM took {time.time() - start_time} seconds")
        # print(f"{all_masks[0].shape = }")
        
        self.scheduled_masks = all_masks    
    
    def generate_all_sam_masks_old(self, image_tensor: torch.Tensor):
        """
        image_tensor: [B, H, W, 3] floats in [0,1]
        Produces self.scheduled_masks as a list of 2D masks (H×W), filtered by size.
        """
        start = time.time()
        device = next(self.sam_model.parameters()).device
        B, H, W, C = image_tensor.shape

        # move channels first and scale to [0,255]
        imgs = (image_tensor.permute(0,3,1,2)).to(device)
        print(f"{imgs.shape = }")

        # tokenize only the images (no prompts)
        inputs = self.processor(images=imgs[0], return_tensors="pt").to(device)
        print(f"{inputs = }")
        with torch.no_grad():
            outputs = self.sam_model(**inputs)

        # outputs.pred_masks: [B, num_masks, h', w']
        # up‐sample back to original H×W
        masks_lowres = outputs.pred_masks    # floats [0,1]
        print(f"{masks_lowres.shape = }")
        print(f"{H = }, {W = }")
        print(f"{masks_lowres[0][0] = }")
        #import pdb; pdb.set_trace()
        masks_resized = F.interpolate(
            masks_lowres,
            size=(H, W),
            mode="bilinear",
            align_corners=False
        )

        # binarize (you can tweak thresh)
        binary_masks = (masks_resized > 0.5).float()  # [B, N, H, W]

        # now filter by area on‐device
        min_area = H * W * 0.005
        max_area = H * W * 0.1
        all_masks = []
        for b in range(B):
            for n in range(binary_masks.shape[1]):
                seg = binary_masks[b, n]            # [H,W]
                area = seg.sum()
                if area < min_area or area > max_area:
                    continue
                all_masks.append(seg)               # still on GPU

        print(f"SAM (HF) took {time.time() - start :.3f}s")
        self.scheduled_masks = all_masks

    def generate_all_sam_masks(self, image_tensor: torch.Tensor):
        B, H, W, C = image_tensor.shape
        max_side = 256
        min_frac, max_frac = 0.1, 0.35

        all_masks = []
        for i in range(B):
            img_orig = (image_tensor[i].detach().cpu().numpy() * 255).astype(np.uint8)
            scale = min(max_side / max(H, W), 1.0)
            h_lr, w_lr = int(H*scale), int(W*scale)
            img_lr = cv2.resize(img_orig, (w_lr, h_lr), interpolation=cv2.INTER_LINEAR)
            print(f"{scale = }")
            print(f"{H = }, {W = }")
            print(f"{h_lr = }, {w_lr = }")

            start = time.time()
            masks_info = self.mask_generator.generate(img_lr)
            print(f"generation took {time.time() - start}")
            print(f"{len(masks_info) = }")

            start = time.time()
            for m in masks_info:
                seg_lr = m["segmentation"]
                frac = seg_lr.sum() / (h_lr * w_lr)
                if not (min_frac <= frac < max_frac):
                    continue
                    
                seg_up = cv2.resize(
                    seg_lr.astype(np.uint8),
                    (W, H), 
                    interpolation=cv2.INTER_NEAREST
                ).astype(bool)

                mask_tensor = torch.from_numpy(seg_up.astype(np.float32)).to(image_tensor.device)
                print(f"post processing {time.time() - start}")
                all_masks.append((mask_tensor, i) if self.masking_each_own else mask_tensor)

        self.scheduled_masks = all_masks


    def save_gif_to_file(self,images, output_file):  
        with io.BytesIO() as writer:  
            images[0].save(  
                writer, format="GIF", save_all=True, append_images=images[1:], duration=100, loop=0  
            )  
            writer.seek(0)  
            with open(output_file, 'wb') as file:  
                file.write(writer.read())
    
    def shape(self):

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        xm = load_model('transmitter', device=device)
        model = load_model('text300M', device=device)
        model.load_state_dict(torch.load('./load/shapE_finetuned_with_330kdata.pth', map_location=device)['model_state_dict'])
        diffusion = diffusion_from_config_shape(load_config('diffusion'))

        batch_size = 1
        guidance_scale = 15.0
        prompt = str(self.cfg.prompt_processor.prompt)
        print('prompt',prompt)

        latents = sample_latents(
            batch_size=batch_size,
            model=model,
            diffusion=diffusion,
            guidance_scale=guidance_scale,
            model_kwargs=dict(texts=[prompt] * batch_size),
            progress=True,
            clip_denoised=True,
            use_fp16=True,
            use_karras=True,
            karras_steps=64,
            sigma_min=1e-3,
            sigma_max=160,
            s_churn=0,
        )
        render_mode = 'nerf' # you can change this to 'stf'
        size = 256 # this is the size of the renders; higher values take longer to render.

        cameras = create_pan_cameras(size, device)

        self.shapeimages = decode_latent_images(xm, latents[0], cameras, rendering_mode=render_mode)

        pc = decode_latent_mesh(xm, latents[0]).tri_mesh()

        skip = 4
        coords = pc.verts
        rgb = np.concatenate([pc.vertex_channels['R'][:,None],pc.vertex_channels['G'][:,None],pc.vertex_channels['B'][:,None]],axis=1) 

        coords = coords[::skip]
        rgb = rgb[::skip]

        self.num_pts = coords.shape[0]
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(coords)
        point_cloud.colors = o3d.utility.Vector3dVector(rgb)
        self.point_cloud = point_cloud

        return coords,rgb,0.4
    
    def add_points(self,coords,rgb):
        pcd_by3d = o3d.geometry.PointCloud()
        pcd_by3d.points = o3d.utility.Vector3dVector(np.array(coords))
        
        bbox = pcd_by3d.get_axis_aligned_bounding_box()
        np.random.seed(0)

        num_points = 1000000  
        points = np.random.uniform(low=np.asarray(bbox.min_bound), high=np.asarray(bbox.max_bound), size=(num_points, 3))

        kdtree = o3d.geometry.KDTreeFlann(pcd_by3d)

        points_inside = []
        color_inside= []
        for point in points:
            _, idx, _ = kdtree.search_knn_vector_3d(point, 1)
            nearest_point = np.asarray(pcd_by3d.points)[idx[0]]
            if np.linalg.norm(point - nearest_point) < 0.01:  # 这个阈值可能需要调整
                points_inside.append(point)
                color_inside.append(rgb[idx[0]]+0.2*np.random.random(3))

        all_coords = np.array(points_inside)
        all_rgb = np.array(color_inside)
        all_coords = np.concatenate([all_coords,coords],axis=0)
        all_rgb = np.concatenate([all_rgb,rgb],axis=0)
        return all_coords,all_rgb
    
    def pcb(self):
        # Since this data set has no colmap data, we start with random points

        # coords,rgb,scale = self.shape()
        # bound= self.radius*scale
        # all_coords,all_rgb = self.add_points(coords,rgb)
        # pcd = BasicPointCloud(points=all_coords *bound, colors=all_rgb, normals=np.zeros((self.num_pts, 3)))

        points = self.skel.sample_smplx_points(N=self.cfg.pts_num)
        pcd = None
        if self.precolor is not None:
            colors = self.precolor
            pcd = BasicPointCloud(self.sampled_points_tensor.detach().cpu().numpy(), colors.detach().cpu().numpy(), None)
        else:
            colors = np.ones_like(points) * 0.5
            pcd = BasicPointCloud(points, colors, None)

        return pcd

    
    def forward(self, batch: Dict[str, Any],renderbackground = None) -> Dict[str, Any]:

        if renderbackground is None:
            renderbackground = self.background_tensor
            
        images = []
        depths = []
        pose_images = []
        self.viewspace_point_list = []

        cycle = self.true_global_step % self.cycle_len
        prompt = self.seg_order[cycle]
        segs = self.segs[prompt]

        if self.true_global_step > 5000:
            prompt = "full"

        if prompt != "full":
            points = self.gaussian.get_xyz.detach()
            mapping_dist, mapping_face, mapping_uvw = self.BVH.signed_distance(
                points, return_uvw=True, mode="raystab"
            )

        for id in range(batch['c2w'].shape[0]):
       
            viewpoint_cam  = Camera(c2w = batch['c2w'][id],FoVy = batch['fovy'][id],height = batch['height'],width = batch['width'])

            if prompt != "full":
                idxs = set().union(*[self.vertexmap[seg] for seg in segs])

                faces = self.skel.faces[mapping_face.cpu()]
                v0, v1, v2 = faces[:, 0], faces[:, 1], faces[:, 2]
                is_in = torch.tensor(
                    [
                        (int(a) in idxs or int(b) in idxs or int(c) in idxs)
                        for a, b, c in zip(v0, v1, v2)
                    ],
                    dtype=torch.bool
                )

                gaussian_idxs = torch.nonzero(is_in, as_tuple=False).squeeze(1)

                render_pkg = render(viewpoint_cam, self.gaussian, self.pipe, renderbackground, subset_ids=gaussian_idxs)
            else:
                render_pkg = render(viewpoint_cam, self.gaussian, self.pipe, renderbackground)
            image, viewspace_point_tensor, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["radii"]
            self.viewspace_point_list.append(viewspace_point_tensor)

            if id == 0:
                self.radii = radii
            else:
                self.radii = torch.max(radii, self.radii)
                
            depth = render_pkg["depth_3dgs"]

            # import kiui
            # kiui.vis.plot_image(image)

            depth = depth.permute(1, 2, 0)
            image = image.permute(1, 2, 0)
            images.append(image)
            depths.append(depth)

            if self.texture_structure_joint:
                backview = abs(batch['azimuth'][id]) > 120 * np.pi / 180
                mvp = batch['mvp_mtx'][id].detach().cpu().numpy()  # [4, 4]
                pose_image, _ = self.skel.humansd_draw(mvp, 512, 512, backview) # [512, 512, 3], fixed pose image resolution
                # kiui.vis.plot_image(pose_image)
                pose_image = torch.from_numpy(pose_image).to(self.device) # [H, W, 3]
                pose_images.append(pose_image)
            else:
                # render pose image
                backview = abs(batch['azimuth'][id]) > 120 * np.pi / 180
                mvp = batch['mvp_mtx'][id].detach().cpu().numpy()  # [4, 4]
                pose_image, _ = self.skel.draw(mvp, 512, 512, backview) # [512, 512, 3], fixed pose image resolution
                # kiui.vis.plot_image(pose_image)
                pose_image = torch.from_numpy(pose_image).to(self.device) # [H, W, 3]
                pose_images.append(pose_image)

            """
            head_idx = set(self.vertexmap["head"])

            faces = self.skel.faces[mapping_face.cpu()]

            v0, v1, v2 = faces[:, 0], faces[:, 1], faces[:, 2]
            is_head = torch.tensor(
                [(int(a) in head_idx or int(b) in head_idx or int(c) in head_idx)
                for a, b, c in zip(v0, v1, v2)],
                dtype=torch.bool,
            )

            # Step 4: Get the indices of Gaussians belonging to head
            head_gaussian_indices = torch.nonzero(is_head, as_tuple=False).squeeze(1)


            render_pkg = render(viewpoint_cam, self.gaussian, self.pipe, renderbackground, subset_ids=head_gaussian_indices)
            image, viewspace_point_tensor, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["radii"]
            self.viewspace_point_list.append(viewspace_point_tensor)

            # manually accumulate max radii across batch
            if id == 0:
                self.radii = radii
            else:
                self.radii = torch.max(radii, self.radii)
                
            depth = render_pkg["depth_3dgs"]

            # import kiui
            # kiui.vis.plot_image(image)

            depth = depth.permute(1, 2, 0)
            image = image.permute(1, 2, 0)
            images.append(image)
            depths.append(depth)
            
            if self.texture_structure_joint:
                backview = abs(batch['azimuth'][id]) > 120 * np.pi / 180
                mvp = batch['mvp_mtx'][id].detach().cpu().numpy()  # [4, 4]
                pose_image, _ = self.skel.humansd_draw(mvp, 512, 512, backview) # [512, 512, 3], fixed pose image resolution
                # kiui.vis.plot_image(pose_image)
                pose_image = torch.from_numpy(pose_image).to(self.device) # [H, W, 3]
                pose_images.append(pose_image)
            else:
                # render pose image
                backview = abs(batch['azimuth'][id]) > 120 * np.pi / 180
                mvp = batch['mvp_mtx'][id].detach().cpu().numpy()  # [4, 4]
                pose_image, _ = self.skel.draw(mvp, 512, 512, backview) # [512, 512, 3], fixed pose image resolution
                # kiui.vis.plot_image(pose_image)
                pose_image = torch.from_numpy(pose_image).to(self.device) # [H, W, 3]
                pose_images.append(pose_image)
            """

        images = torch.stack(images, 0)
        depths = torch.stack(depths, 0)
        pose_images = torch.stack(pose_images, 0)

        self.visibility_filter = self.radii > 0.0

        # mask near-hand points from visibility_filter, since we don't want to densify them
        if self.cfg.disable_hand_densification:
            points = self.gaussian.get_xyz # [N, 3]
            hand_centers = torch.from_numpy(self.skel.hand_centers).to(points.dtype).to(points.device) # [2, 3]
            distance = torch.norm(points[:, None, :] - hand_centers[None, :, :], dim=-1) # [N, 2]
            hand_mask = distance.min(dim=-1).values < self.cfg.hand_radius # [N]
            self.visibility_filter = self.visibility_filter & (~hand_mask)

        render_pkg["comp_rgb"] = images
        render_pkg["depth"] = depths
        render_pkg['pose'] = pose_images
        render_pkg["opacity"] = depths / (depths.max() + 1e-5)

        return {
            **render_pkg,
        }

    def on_fit_start(self) -> None:
        super().on_fit_start()
        # only used in training
        self.prompt_processor = threestudio.find(self.cfg.prompt_processor_type)(
            self.cfg.prompt_processor
        )

        self.prompt_processor.prepare_prompt_options_embeddings(self.prompt_options_json)
        self.guidance = threestudio.find(self.cfg.guidance_type)(self.cfg.guidance)

    
    def training_step(self, batch, batch_idx):

        self.gaussian.update_learning_rate(self.true_global_step)
        
        if self.true_global_step > self.cfg.half_scheduler_max_step:
            self.guidance.set_min_max_steps(min_step_percent=0.02, max_step_percent=0.55)

        self.gaussian.update_learning_rate(self.true_global_step)

        out = self(batch) 

        cycle = self.true_global_step % self.cycle_len
        prompt = self.seg_order[cycle]
        if self.true_global_step > 5000:
            prompt = "full"

        prompt_utils = self.prompt_processor(prompt)
        images = out["comp_rgb"]
        depth_images = out['depth']
        depth_min = torch.amin(depth_images, dim=[1, 2, 3], keepdim=True)
        depth_max = torch.amax(depth_images, dim=[1, 2, 3], keepdim=True)
        depth_images = (depth_images - depth_min) / (depth_max - depth_min + 1e-10)# to [0, 1]
        depth_images = depth_images.repeat(1, 1, 1, 3)# to 3-channel
        control_images = out['pose']

        # guidance_eval = (self.true_global_step % 200 == 0)
        guidance_eval = False
        
        if self.texture_structure_joint:
            guidance_out = self.guidance(
                control_images, images, depth_images, prompt_utils, **batch, 
                rgb_as_latents=False, guidance_eval=guidance_eval
            )
        elif self.controlnet:
            guidance_out = self.guidance(
                control_images, images, prompt_utils, **batch, 
                rgb_as_latents=False, guidance_eval=guidance_eval
            )
        else:
            guidance_out = self.guidance(
                images, prompt_utils, **batch, 
                rgb_as_latents=False, guidance_eval=guidance_eval
            )

        loss = 0.0

        loss = loss + guidance_out['loss_sds'] *self.C(self.cfg.loss['lambda_sds'])
        
        loss_sparsity = (out["opacity"] ** 2 + 0.01).sqrt().mean()
        self.log("train/loss_sparsity", loss_sparsity)
        loss += loss_sparsity * self.C(self.cfg.loss.lambda_sparsity)

        opacity_clamped = out["opacity"].clamp(1.0e-3, 1.0 - 1.0e-3)
        loss_opaque = binary_cross_entropy(opacity_clamped, opacity_clamped)
        self.log("train/loss_opaque", loss_opaque)
        loss += loss_opaque * self.C(self.cfg.loss.lambda_opaque)

        if not self.masking:
            print("We are not masking")
            if guidance_eval:
                self.guidance_evaluation_save(
                    out["comp_rgb"].detach()[: guidance_out["eval"]["bs"]],
                    guidance_out["eval"],
                )

        elif self.masking_each_own and self.global_step % 51 == 0:
            print("We ARE masking")
            start = time.time()
            self.generate_all_sam_masks(images)
            print(f"generation took {time.time() - start}")
            
            fvecs = []
            start = time.time()
            for seg, img_idx in self.scheduled_masks:
                # build the single masked crop
                m = seg.unsqueeze(-1)                              # [H,W,1]
                crop = (images[img_idx] * m).unsqueeze(0)           # [1,H,W,3]
                crop_chw = crop.permute(0,3,1,2)                    # [1,3,H,W]
                # resize + extract features
                x224 = F.interpolate(crop_chw, (224,224), mode="bilinear", align_corners=False)
                with torch.no_grad():
                    fmap = self.feature_extractor(x224)             # [1,C,h,w]
                fvec = fmap.mean(dim=[2,3]).squeeze(0)              # [C]
                fvecs.append(fvec)
            if len(fvecs) > 0:
                Ff = torch.stack(fvecs, dim=0)
                F_norm = Ff / Ff.norm(dim=1, keepdim=True)
                print(f"features {time.time() - start}")

                available = list(range(len(self.scheduled_masks)))  
                used = set()
                total_calls = 4  

                start = time.time()

                for _call_i in range(total_calls):  
                    # pick from the ones never used yet  
                    avail = [i for i in available if i not in used]  
                    if len(avail) < 1:  
                        break  

                    # 1) random anchor  
                    anchor_idx = random.choice(avail)  
                    used.add(anchor_idx)  

                    # 2) find its 3 most similar among the remaining pool  
                    sims = (F_norm[anchor_idx:anchor_idx+1] @ F_norm.T).squeeze(0)  
                    # make sure sims for already-used are -inf so topk skips them  
                    min_val = torch.finfo(sims.dtype).min
                    for u in used:  
                        sims[u] = min_val  
                    k = min(3, len(avail) - 1)  
                    if k > 0:  
                        topk = sims.topk(k=k).indices.tolist()  
                        for t in topk:  
                            used.add(t)  
                        selected = [anchor_idx] + topk  
                    else:  
                        selected = [anchor_idx]  

                    # 3) build your mini-batch of masked RGB/depth/ctrl  
                    rgb_list, d_list, ctrl_list = [], [], []  
                    batch_info = {k: [] for k,v in batch.items() if torch.is_tensor(v) and v.shape[0] == images.shape[0]}  
                    for idx in selected:  
                        seg, img_idx = self.scheduled_masks[idx]  
                        m = seg.unsqueeze(-1)  
                        rgb_list.append((images[img_idx]*m).unsqueeze(0))  
                        d_list.append((depth_images[img_idx]*m).unsqueeze(0))  
                        ctrl_list.append(control_images[img_idx:img_idx+1])  
                        for k in batch_info:  
                            batch_info[k].append(batch[k][img_idx:img_idx+1])  

                    mi_cat  = torch.cat(rgb_list, dim=0)  
                    md_cat  = torch.cat(d_list, dim=0)  
                    ctrl_cat= torch.cat(ctrl_list, dim=0)

                    # mi_vis = mi_cat.permute(0, 3, 1, 2).detach().cpu()

                    # # make a grid of all N masked crops on one row
                    # grid = make_grid(mi_vis, nrow=mi_vis.size(0), normalize=True, value_range=(0,1))

                    # # choose an output directory (or reuse your existing `out_dir`)
                    # out_dir = "/n/holylfs05/LABS/pfister_lab/Lab/coxfs01/pfister_lab2/Lab/abenahmedk/interm_vis"
                    # os.makedirs(out_dir, exist_ok=True)

                    # # save it
                    # save_image(
                    #     grid,
                    #     os.path.join(out_dir, f"selected_masks_step{self.global_step}.png")
                    # )
                    # print(f"🔍 Saved selected masked crops to {out_dir}/selected_masks_step{self.global_step}.png")  
                    for k in batch_info:  
                        batch_info[k] = torch.cat(batch_info[k], dim=0)  
                    # print(f"{ctrl_cat = }")
                    # # 4) single SDS call  
                    # guidance_out = self.guidance(  
                    #     mi_cat, md_cat, prompt_utils=prompt_utils,  
                    #     control_images=ctrl_cat,  
                    #     **batch_info, rgb_as_latents=False  
                    # )  
                    if self.texture_structure_joint:
                        # (control, rgb, depth, prompt)
                        guidance_out = self.guidance(
                            ctrl_cat,         # control_images
                            mi_cat,           # rgb
                            md_cat,           # depth
                            prompt_utils,     # prompt
                            **batch_info,
                            rgb_as_latents=False,
                        )
                    elif self.controlnet:
                        # (control, rgb, prompt)
                        guidance_out = self.guidance(
                            ctrl_cat,         # control_images
                            mi_cat,           # rgb
                            prompt_utils,     # prompt
                            **batch_info,
                            rgb_as_latents=False,
                        )
                    else:
                        # (rgb, prompt)
                        guidance_out = self.guidance(
                            mi_cat,           # rgb
                            prompt_utils,     # prompt
                            **batch_info,
                            rgb_as_latents=False,
                        )
                    loss += guidance_out['loss_sds'].mean() * self.C(self.cfg.loss['lambda_sds'])  

            print(f"rest took {time.time() - start}")
        for name, value in self.cfg.loss.items():
            self.log(f"train_params/{name}", self.C(value))
        return {"loss": loss}

    def on_before_optimizer_step(self, optimizer):

        # return

        with torch.no_grad():
            
            if self.true_global_step < self.cfg.densify_prune_end_step: # 15000
                viewspace_point_tensor_grad = torch.zeros_like(self.viewspace_point_list[0])
                for idx in range(len(self.viewspace_point_list)):
                    viewspace_point_tensor_grad = viewspace_point_tensor_grad + self.viewspace_point_list[idx].grad
                # Keep track of max radii in image-space for pruning
                self.gaussian.max_radii2D[self.visibility_filter] = torch.max(self.gaussian.max_radii2D[self.visibility_filter], self.radii[self.visibility_filter])
                
                self.gaussian.add_densification_stats(viewspace_point_tensor_grad, self.visibility_filter)

                if self.true_global_step > self.cfg.densify_prune_start_step and self.true_global_step % self.cfg.densify_prune_interval == 0: # 500 100
                    size_threshold = self.cfg.size_threshold if self.true_global_step > self.cfg.size_threshold_fix_step else None # 3000
                    self.gaussian.densify_and_prune(self.cfg.max_grad , 0.05, self.cameras_extent, size_threshold) 

            # prune-only phase according to Gaussian size, rather than the stochastic gradient to eliminate floating artifacts.
            if self.true_global_step > self.cfg.prune_only_start_step and self.true_global_step < self.cfg.prune_only_end_step:
                viewspace_point_tensor_grad = torch.zeros_like(self.viewspace_point_list[0])
                for idx in range(len(self.viewspace_point_list)):
                    viewspace_point_tensor_grad = viewspace_point_tensor_grad + self.viewspace_point_list[idx].grad
                # Keep track of max radii in image-space for pruning
                self.gaussian.max_radii2D[self.visibility_filter] = torch.max(self.gaussian.max_radii2D[self.visibility_filter], self.radii[self.visibility_filter])
                
                self.gaussian.add_densification_stats(viewspace_point_tensor_grad, self.visibility_filter)

                if self.true_global_step % self.cfg.prune_only_interval == 0:
                    self.gaussian.prune_only(size_thresh=self.cfg.prune_size_threshold)

    def validation_step(self, batch, batch_idx):
        out = self(batch)
        self.save_image_grid(
            f"it{self.true_global_step}-{batch['index'][0]}.png",
            (
                [
                    {
                        "type": "rgb",
                        "img": batch["rgb"][0],
                        "kwargs": {"data_format": "HWC"},
                    }
                ]
                if "rgb" in batch
                else []
            )
            + [
                {
                    "type": "rgb",
                    "img": out["comp_rgb"][0],
                    "kwargs": {"data_format": "HWC"},
                },
            ]
            + (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_normal"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "comp_normal" in out
                else []
            ),
            name="validation_step",
            step=self.true_global_step,
        )
        # save_path = self.get_save_path(f"it{self.true_global_step}-val.ply")
        # self.gaussian.save_ply(save_path)
        # load_ply(save_path,self.get_save_path(f"it{self.true_global_step}-val-color.ply"))

    def on_validation_epoch_end(self):
        pass

    def test_step(self, batch, batch_idx):
        only_rgb = True
        bg_color = [1, 1, 1] if self.cfg.bg_white else [0, 0, 0]

        testbackground_tensor = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        out = self(batch,testbackground_tensor)
        if only_rgb:
            self.save_image_grid(
                f"it{self.true_global_step}-test/{batch['index'][0]}.png",
                (
                    [
                        {
                            "type": "rgb",
                            "img": batch["rgb"][0],
                            "kwargs": {"data_format": "HWC"},
                        }
                    ]
                    if "rgb" in batch
                    else []
                )
                + [
                    {
                        "type": "rgb",
                        "img": out["comp_rgb"][0],
                        "kwargs": {"data_format": "HWC"},
                    },
                ]
                + (
                    [
                        {
                            "type": "rgb",
                            "img": out["comp_normal"][0],
                            "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                        }
                    ]
                    if "comp_normal" in out
                    else []
                ),
                name="test_step",
                step=self.true_global_step,
            )
        else:
            self.save_image_grid(
                f"it{self.true_global_step}-test/{batch['index'][0]}.png",
                (
                    [
                        {
                            "type": "rgb",
                            "img": batch["rgb"][0],
                            "kwargs": {"data_format": "HWC"},
                        }
                    ]
                    if "rgb" in batch
                    else []
                )
                + [
                    {
                        "type": "rgb",
                        "img": out["comp_rgb"][0],
                        "kwargs": {"data_format": "HWC"},
                    },
                ]
                + (
                    [
                        {
                            "type": "rgb",
                            "img": out["comp_normal"][0],
                            "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                        }
                    ]
                    if "comp_normal" in out
                    else []
                )
                + (
                    [
                        {
                            "type": "grayscale",
                            "img": out["depth"][0],
                            "kwargs": {},
                        }
                    ]
                    if "depth" in out
                    else []
                )
                + [
                    {
                        "type": "grayscale",
                        "img": out["opacity"][0, :, :, 0],
                        "kwargs": {"cmap": None, "data_range": (0, 1)},
                    },
                ],
                name="test_step",
                step=self.true_global_step,
            )


    def on_test_epoch_end(self):
        self.save_img_sequence(
            f"it{self.true_global_step}-test",
            f"it{self.true_global_step}-test",
            "(\d+)\.png",
            save_format="mp4",
            fps=30,
            name="test",
            step=self.true_global_step,
        )
        save_path = self.get_save_path(f"last.ply")
        self.gaussian.save_ply(save_path)
        # self.pointefig.savefig(self.get_save_path("pointe.png"))
        # o3d.io.write_point_cloud(self.get_save_path("shape.ply"), self.point_cloud)
        # self.save_gif_to_file(self.shapeimages, self.get_save_path("shape.gif"))
        # load_ply(save_path,self.get_save_path(f"it{self.true_global_step}-test-color.ply"))
        

    def configure_optimizers(self):
        opt = OptimizationParams(self.parser)

        point_cloud = self.pcb()
        self.gaussian.create_from_pcd(point_cloud, self.cameras_extent)

        """
        self.skel.normalize()
        dev = self.gaussian._xyz.device

        center = torch.tensor(self.skel.ori_center, device=dev)
        scale = torch.tensor(self.skel.ori_scale, device=dev)

        pos = self.gaussian._xyz.detach().clone()
        pos = (pos - center) * scale
        pos[:, [1, 2]] = pos[:, [2, 1]]
        pos *= (1.1 ** 10)

        self.skel.scale(-10)

        self.gaussian._xyz = pos.clone().requires_grad_()
        """
        self.gaussian.training_setup(opt)

        """
        if self.use_img:
            
            print(f"Extrinsic: {self.extrinsic[0]}")
            print(f"Intrinsic: {self.intrinsic[0]}")

            self.gaussian.bake_colors_from_image_and_camera(self.image, self.intrinsic, self.extrinsic, self.image.shape[1:3])
           
            fy = self.intrinsic[0][1, 1]
            H = self.image.shape[1]
            W = self.image.shape[2]
            FoVy = 2 * math.atan(0.5 * H / fy)
           
            cam = Camera(self.extrinsic, FoVy, H, W)
           
            rendered = render(cam, self.gaussian, self.pipe, bg_color=torch.tensor([0.0, 0.0, 0.0], device="cuda"))
           
            print("Render shape: ", rendered["render"].shape)
           
            img_out = rendered["render"].clamp(0, 1).cpu()
           
            save_image(img_out, self.get_save_path(f"cameraview.png"))
           
            debug_render_splats_and_camera(
                self.gaussian,
                torch.inverse(self.extrinsic),
                self.intrinsic[0],
                self.image.shape[1:3],
                save_path=self.get_save_path(f"debug_camera_view.png")
            )
            """
        ret = {
            "optimizer": self.gaussian.optimizer,
        }

        return ret
    
    def guidance_evaluation_save(self, comp_rgb, guidance_eval_out):
        B, size = comp_rgb.shape[:2]
        resize = lambda x: F.interpolate(
            x.permute(0, 3, 1, 2), (size, size), mode="bilinear", align_corners=False
        ).permute(0, 2, 3, 1)
        filename = f"it{self.true_global_step}-train.png"

        def merge12(x):
            return x.reshape(-1, *x.shape[2:])

        self.save_image_grid(
            filename,
            [
                {
                    "type": "rgb",
                    "img": merge12(comp_rgb),
                    "kwargs": {"data_format": "HWC"},
                },
            ]
            + (
                [
                    {
                        "type": "rgb",
                        "img": merge12(resize(guidance_eval_out["imgs_noisy"])),
                        "kwargs": {"data_format": "HWC"},
                    }
                ]
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": merge12(resize(guidance_eval_out["imgs_1step"])),
                        "kwargs": {"data_format": "HWC"},
                    }
                ]
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": merge12(resize(guidance_eval_out["imgs_1orig"])),
                        "kwargs": {"data_format": "HWC"},
                    }
                ]
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": merge12(resize(guidance_eval_out["imgs_final"])),
                        "kwargs": {"data_format": "HWC"},
                    }
                ]
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": merge12(resize(guidance_eval_out["midas_depth_imgs_noisy"])),
                        "kwargs": {"data_format": "HWC"},
                    }
                ]
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": merge12(resize(guidance_eval_out["midas_depth_imgs_1step"])),
                        "kwargs": {"data_format": "HWC"},
                    }
                ]
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": merge12(resize(guidance_eval_out["midas_depth_imgs_1orig"])),
                        "kwargs": {"data_format": "HWC"},
                    }
                ]
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": merge12(resize(guidance_eval_out["midas_depth_imgs_final"])),
                        "kwargs": {"data_format": "HWC"},
                    }
                ]
            ),
            name="train_step",
            step=self.true_global_step,
            texts=guidance_eval_out["texts"],
        )