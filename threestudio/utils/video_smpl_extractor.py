"""
Video SMPL Pose Extraction Utility

This module provides functionality to extract SMPL-X poses from video files
using the NLF (Neural Localizer Fields) model. It processes videos frame-by-frame,
extracts poses for all frames, and saves them in a format compatible with
the animation pipeline.
"""

import os
import numpy as np
import torch
import torchvision
import torchvision.transforms.functional as TFunc
from typing import Dict, Tuple
import cv2
from tqdm import tqdm


class VideoSMPLExtractor:
    """
    Extracts SMPL-X pose parameters from video files using NLF model.

    The extractor:
    1. Loads the first frame for initial pose estimation and training
    2. Processes all subsequent frames to extract SMPL-X parameters
    3. Saves pose sequences in .npz format for animation
    """

    def __init__(self, model_path: str = 'nlf/models/nlf_l_multi.torchscript'):
        """
        Initialize the video SMPL extractor.

        Args:
            model_path: Path to the NLF TorchScript model
        """
        self.model_path = model_path
        self.model = None

    def load_model(self):
        """Load the NLF model onto GPU."""
        if self.model is None:
            print(f"Loading NLF model from {self.model_path}...")
            self.model = torch.jit.load(self.model_path).cuda().eval()
            print("NLF model loaded successfully.")

    def extract_frames_from_video(self, video_path: str) -> Tuple[torch.Tensor, int, int]:
        """
        Extract all frames from a video file.

        Args:
            video_path: Path to the input video file

        Returns:
            frames: Tensor of shape (num_frames, C, H, W)
            fps: Frame rate of the video
            num_frames: Total number of frames
        """
        print(f"Extracting frames from video: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"Video info: {num_frames} frames, {fps} FPS, {width}x{height}")

        frames = []
        for _ in tqdm(range(num_frames), desc="Reading frames"):
            ret, frame = cap.read()
            if not ret:
                break
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Convert to torch tensor (H, W, C) -> (C, H, W)
            frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float()
            frames.append(frame_tensor)

        cap.release()

        if len(frames) == 0:
            raise ValueError("No frames were extracted from the video")

        frames_tensor = torch.stack(frames)
        print(f"Extracted {len(frames)} frames with shape: {frames_tensor.shape}")

        return frames_tensor, fps, len(frames)

    def process_frame_batch(self, frames: torch.Tensor, batch_size: int = 4) -> Dict[str, torch.Tensor]:
        """
        Process a batch of frames through the NLF model.

        Args:
            frames: Tensor of shape (num_frames, C, H, W)
            batch_size: Number of frames to process at once

        Returns:
            Dictionary containing SMPL-X parameters for all frames:
                - pose: (num_frames, 165) - SMPL-X pose parameters
                - betas: (num_frames, 10) - Shape parameters
                - trans: (num_frames, 3) - Translation parameters
                - vertices3d: (num_frames, num_verts, 3) - 3D vertices
        """
        self.load_model()

        num_frames = frames.shape[0]
        all_results = {
            'pose': [],
            'betas': [],
            'trans': [],
            'vertices3d': []
        }

        print(f"Processing {num_frames} frames in batches of {batch_size}...")

        with torch.inference_mode(), torch.device('cuda'):
            for i in tqdm(range(0, num_frames, batch_size), desc="Processing batches"):
                batch_end = min(i + batch_size, num_frames)
                frame_batch = frames[i:batch_end].cuda()

                # Run NLF detection
                pred = self.model.detect_smpl_batched(frame_batch, model_name='smplx')

                # Handle both list and tensor outputs
                # NLF returns lists when processing batches
                for key in ['pose', 'betas', 'trans', 'vertices3d']:
                    if key in pred:
                        value = pred[key]
                        # Convert list to tensor if needed
                        if isinstance(value, list):
                            value = torch.stack([torch.tensor(v) if not isinstance(v, torch.Tensor) else v for v in value])
                        # Move to CPU
                        all_results[key].append(value.cpu())

        # Concatenate all batches
        for key in all_results:
            if len(all_results[key]) > 0:
                all_results[key] = torch.cat(all_results[key], dim=0)

        print(f"Processed all frames. Pose shape: {all_results['pose'].shape}")
        return all_results

    def get_first_frame_data(self, frames: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract SMPL-X data from the first frame for initialization.

        Args:
            frames: Tensor of shape (num_frames, C, H, W)

        Returns:
            Dictionary with first frame SMPL-X parameters
        """
        self.load_model()

        # Apply same transformations as in GaussianDreamer.py
        first_frame = TFunc.rotate(TFunc.hflip(frames[0]), angle=180).cuda()
        frame_batch = first_frame.unsqueeze(0)

        with torch.inference_mode(), torch.device('cuda'):
            pred = self.model.detect_smpl_batched(frame_batch, model_name='smplx')

        # Extract single frame results
        # Handle both list and tensor outputs from NLF
        result = {}
        for key in ['pose', 'betas', 'trans', 'vertices3d']:
            if key in pred:
                value = pred[key]
                # If it's a list, get first element
                if isinstance(value, list):
                    value = value[0]
                    # Convert to tensor if needed
                    if not isinstance(value, torch.Tensor):
                        value = torch.tensor(value)
                else:
                    # If it's a tensor, index it
                    value = value[0]
                result[key] = value

        return result

    def save_initial_pose(self,
                         first_frame_smpl: Dict[str, torch.Tensor],
                         output_path: str):
        """
        Save initial pose from first frame for animation.py binding.

        Args:
            first_frame_smpl: Dictionary with first frame SMPL-X parameters
            output_path: Path to save the .npz file
        """
        pose = first_frame_smpl['pose']  # (165,)

        # Extract body pose (21 joints)
        body_pose_flat = pose[3:3+21*3]  # (63,)
        body_pose = body_pose_flat.reshape(21, 3)  # (21, 3)

        # Extract hand poses
        left_hand_pose_flat = pose[25*3:40*3]  # (45,)
        left_hand_pose = left_hand_pose_flat.reshape(15, 3)  # (15, 3)

        right_hand_pose_flat = pose[40*3:55*3]  # (45,)
        right_hand_pose = right_hand_pose_flat.reshape(15, 3)  # (15, 3)

        save_dict = {
            'body_pose': body_pose.cpu().numpy(),
            'left_hand_pose': left_hand_pose.cpu().numpy(),
            'right_hand_pose': right_hand_pose.cpu().numpy(),
            'betas': first_frame_smpl['betas'].cpu().numpy(),
            'trans': first_frame_smpl['trans'].cpu().numpy(),
        }

        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        np.savez(output_path, **save_dict)
        print(f"Saved initial pose to: {output_path}")
        print(f"  - Body pose shape: {body_pose.shape}")
        print(f"  - Left hand pose shape: {left_hand_pose.shape}")
        print(f"  - Right hand pose shape: {right_hand_pose.shape}")

    def save_pose_sequence(self,
                          smpl_data: Dict[str, torch.Tensor],
                          output_path: str,
                          fps: int = 30):
        """
        Save SMPL-X pose sequence in format compatible with animation.py.

        The animation.py script expects:
        - 'poses': (num_frames, 21, 3) - body pose in axis-angle format

        Args:
            smpl_data: Dictionary with SMPL-X parameters
            output_path: Path to save the .npz file
            fps: Frame rate of the video
        """
        pose = smpl_data['pose']  # (num_frames, 165)

        # Extract body pose (21 joints)
        # SMPL-X pose format: [global_orient(3), body_pose(63), ...]
        # We need joints 1-22 (body_pose starts at index 3, 21*3=63 params)
        body_pose_flat = pose[:, 3:3+21*3]  # (num_frames, 63)
        body_pose = body_pose_flat.reshape(-1, 21, 3)  # (num_frames, 21, 3)

        # Optionally save other parameters
        save_dict = {
            'poses': body_pose.cpu().numpy(),
            'fps': fps,
            'num_frames': pose.shape[0],
            # Full SMPL-X parameters for potential future use
            'smplx_pose': pose.cpu().numpy(),
            'smplx_betas': smpl_data['betas'].cpu().numpy(),
            'smplx_trans': smpl_data['trans'].cpu().numpy(),
        }

        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        np.savez(output_path, **save_dict)
        print(f"Saved pose sequence to: {output_path}")
        print(f"  - Number of frames: {pose.shape[0]}")
        print(f"  - Body pose shape: {body_pose.shape}")
        print(f"  - FPS: {fps}")

    def process_video(self,
                     video_path: str,
                     output_dir: str,
                     batch_size: int = 4) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], str, str]:
        """
        Complete pipeline: extract frames, process all poses, save sequence.

        Args:
            video_path: Path to input video
            output_dir: Directory to save output files
            batch_size: Batch size for processing

        Returns:
            first_frame: First frame tensor for initialization
            first_frame_smpl: SMPL-X data for first frame
            pose_sequence_path: Path to saved pose sequence .npz file
            initial_pose_path: Path to saved initial pose .npz file
        """
        # Extract frames
        frames, fps, num_frames = self.extract_frames_from_video(video_path)

        # Get first frame data for initialization
        print("\nProcessing first frame for initialization...")
        first_frame_smpl = self.get_first_frame_data(frames)
        first_frame = TFunc.rotate(TFunc.hflip(frames[0]), angle=180)

        # Get video name for file paths
        video_name = os.path.splitext(os.path.basename(video_path))[0]

        # Save initial pose for animation binding
        initial_pose_path = os.path.join(output_dir, f"{video_name}_initial_pose.npz")
        self.save_initial_pose(first_frame_smpl, initial_pose_path)

        # Process all frames
        print("\nProcessing all frames for pose sequence...")
        all_smpl_data = self.process_frame_batch(frames, batch_size=batch_size)

        # Save pose sequence
        pose_sequence_path = os.path.join(output_dir, f"{video_name}_poses.npz")
        self.save_pose_sequence(all_smpl_data, pose_sequence_path, fps=fps)

        return first_frame, first_frame_smpl, pose_sequence_path, initial_pose_path


def extract_smpl_from_video(video_path: str,
                            output_dir: str,
                            batch_size: int = 4) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], str, str]:
    """
    Convenience function to extract SMPL poses from video.

    Args:
        video_path: Path to input video file
        output_dir: Directory to save extracted poses
        batch_size: Batch size for processing frames

    Returns:
        first_frame: First frame tensor for training
        first_frame_smpl: SMPL-X parameters for first frame
        pose_sequence_path: Path to saved pose sequence
        initial_pose_path: Path to saved initial pose
    """
    extractor = VideoSMPLExtractor()
    return extractor.process_video(video_path, output_dir, batch_size)
