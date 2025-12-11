"""
Utility to repose Gaussian Splats from training pose to A-pose.

This is needed for video mode where Gaussians are trained on a specific pose
but need to be saved in the default A-pose for animation.py compatibility.
"""

import os
import numpy as np
import torch
from typing import Dict

# Import from animation.py's rendering system
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from gs_renderer import Renderer


def repose_gaussian_ply(
    input_ply_path: str,
    output_ply_path: str,
    training_pose_path: str,
    smplx_path: str,
    gender: str = 'neutral'
):
    """
    Repose Gaussians from training pose to default A-pose.

    Args:
        input_ply_path: Path to trained .ply file (in training pose)
        output_ply_path: Path to save reposed .ply file (in A-pose)
        training_pose_path: Path to .npz file with training pose parameters
        smplx_path: Path to SMPL-X models directory
        gender: SMPL-X gender ('neutral', 'male', 'female')
    """
    print(f"[Repose] Loading Gaussians from {input_ply_path}")

    # Load Gaussians
    gs = Renderer(sh_degree=0, white_background=False)
    gs.gaussians.load_ply(input_ply_path)

    # Get Gaussian positions
    points = gs.gaussians.get_xyz.detach().cpu().numpy()
    print(f"[Repose] Loaded {len(points)} Gaussians")

    # Load SMPL-X model
    try:
        import smplx
    except ImportError:
        raise ImportError("smplx package required. Install with: pip install smplx")

    smplx_model = smplx.create(
        smplx_path,
        model_type='smplx',
        gender=gender,
        use_face_contour=False,
        num_betas=10,
        num_expression_coeffs=10,
        ext='npz',
        use_pca=False,
    )

    # Load training pose
    print(f"[Repose] Loading training pose from {training_pose_path}")
    training_data = np.load(training_pose_path)
    body_pose = torch.tensor(training_data['body_pose'], dtype=torch.float32).unsqueeze(0)  # (1, 21, 3)
    left_hand_pose = torch.tensor(training_data['left_hand_pose'], dtype=torch.float32).unsqueeze(0)  # (1, 15, 3)
    right_hand_pose = torch.tensor(training_data['right_hand_pose'], dtype=torch.float32).unsqueeze(0)  # (1, 15, 3)
    betas = torch.tensor(training_data['betas'], dtype=torch.float32).unsqueeze(0) if 'betas' in training_data else torch.zeros(1, 10)
    transl = torch.tensor(training_data['trans'], dtype=torch.float32).unsqueeze(0) if 'trans' in training_data else torch.zeros(1, 3)

    # Create SMPL-X mesh in TRAINING pose
    print("[Repose] Creating SMPL-X mesh in training pose")
    output_train = smplx_model(
        betas=betas,
        body_pose=body_pose,
        left_hand_pose=left_hand_pose,
        right_hand_pose=right_hand_pose,
        transl=transl,
        return_verts=True
    )
    vertices_train = output_train.vertices.detach().cpu().numpy()[0]  # (num_verts, 3)
    faces = smplx_model.faces.astype(np.int32)

    print(f"[Repose] SMPL-X mesh: {vertices_train.shape[0]} vertices, {faces.shape[0]} faces")

    # Compute mapping from Gaussians to mesh faces (in training pose)
    print("[Repose] Computing Gaussian-to-mesh mapping...")
    try:
        import cubvh
    except ImportError:
        raise ImportError("cubvh package required for BVH. Install from https://github.com/ashawkey/cubvh")

    vertices_train_torch = torch.tensor(vertices_train, dtype=torch.float32).cuda()
    faces_torch = torch.tensor(faces, dtype=torch.int32).cuda()
    points_torch = torch.tensor(points, dtype=torch.float32).cuda()

    BVH = cubvh.cuBVH(vertices_train_torch, faces_torch)
    mapping_dist, mapping_face, mapping_uvw = BVH.signed_distance(
        points_torch, return_uvw=True, mode="raystab"
    )

    mapping_dist = mapping_dist.detach().cpu().numpy()
    mapping_face = mapping_face.detach().cpu().numpy().astype(np.int32)
    mapping_uvw = mapping_uvw.detach().cpu().numpy().astype(np.float32)

    print(f"[Repose] Mapping computed for {len(mapping_face)} Gaussians")

    # Create default A-pose
    print("[Repose] Creating SMPL-X mesh in default A-pose")
    body_pose_apose = np.zeros((21, 3), dtype=np.float32)
    # Default A-pose from animation.py lines 173-180
    body_pose_apose[15, 2] = -0.7853982  # left_shoulder
    body_pose_apose[16, 2] = 0.7853982   # right_shoulder
    body_pose_apose[0, 1] = 0.2          # left_hip
    body_pose_apose[0, 2] = 0.1
    body_pose_apose[1, 1] = -0.2         # right_hip
    body_pose_apose[1, 2] = -0.1

    body_pose_apose_torch = torch.tensor(body_pose_apose, dtype=torch.float32).unsqueeze(0)
    left_hand_pose_zero = torch.zeros(1, 15, 3, dtype=torch.float32)
    right_hand_pose_zero = torch.zeros(1, 15, 3, dtype=torch.float32)

    output_apose = smplx_model(
        betas=betas,
        body_pose=body_pose_apose_torch,
        left_hand_pose=left_hand_pose_zero,
        right_hand_pose=right_hand_pose_zero,
        transl=transl,
        return_verts=True
    )
    vertices_apose = output_apose.vertices.detach().cpu().numpy()[0]

    # Use mapping to compute new Gaussian positions
    print("[Repose] Computing new Gaussian positions in A-pose...")
    faces_mapped = faces[mapping_face]
    v0 = vertices_apose[faces_mapped[:, 0]]
    v1 = vertices_apose[faces_mapped[:, 1]]
    v2 = vertices_apose[faces_mapped[:, 2]]

    # Compute face normals
    e1 = v1 - v0
    e2 = v2 - v0
    fnormals = np.cross(e1, e2)
    fnormals = fnormals / (np.linalg.norm(fnormals, axis=-1, keepdims=True) + 1e-8)

    # Compute closest points on mesh
    cpoints = (
        v0 * mapping_uvw[:, [0]]
        + v1 * mapping_uvw[:, [1]]
        + v2 * mapping_uvw[:, [2]]
    )

    # New Gaussian positions
    points_apose = cpoints + mapping_dist[:, None] * fnormals

    # Update Gaussian positions
    print("[Repose] Updating Gaussian positions...")
    gs.gaussians._xyz = torch.tensor(points_apose, dtype=torch.float32).cuda()

    # Save reposed Gaussians
    print(f"[Repose] Saving reposed Gaussians to {output_ply_path}")
    os.makedirs(os.path.dirname(output_ply_path) if os.path.dirname(output_ply_path) else '.', exist_ok=True)
    gs.gaussians.save_ply(output_ply_path)

    print("[Repose] Reposing complete!")
    print(f"  Input:  {input_ply_path} (training pose)")
    print(f"  Output: {output_ply_path} (A-pose)")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Repose Gaussian Splats from training pose to A-pose")
    parser.add_argument("--input_ply", type=str, required=True, help="Path to input .ply file (training pose)")
    parser.add_argument("--output_ply", type=str, required=True, help="Path to output .ply file (A-pose)")
    parser.add_argument("--training_pose", type=str, required=True, help="Path to training pose .npz file")
    parser.add_argument("--smplx_path", type=str, required=True, help="Path to SMPL-X models directory")
    parser.add_argument("--gender", type=str, default="neutral", choices=["neutral", "male", "female"],
                        help="SMPL-X gender")

    args = parser.parse_args()

    repose_gaussian_ply(
        input_ply_path=args.input_ply,
        output_ply_path=args.output_ply,
        training_pose_path=args.training_pose,
        smplx_path=args.smplx_path,
        gender=args.gender
    )
