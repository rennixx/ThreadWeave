"""
Animator Module

Creates video clips with camera movements from static images.
"""

import logging
import os
from typing import Tuple

import cv2
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)


class Animator:
    """
    Creates animated video clips from static images with camera movements.

    Supports zoom and pan effects using OpenCV.
    """

    VALID_MOVEMENTS = ["static", "zoom_in", "zoom_out", "pan_left", "pan_right"]

    def __init__(self, fps: int = 30, resolution: Tuple[int, int] = (1080, 1920)):
        """
        Initialize the animator.

        Args:
            fps: Frames per second for output video
            resolution: (width, height) tuple for output resolution
        """
        self.fps = fps
        self.resolution = resolution
        logger.info(f"Animator initialized: {fps} FPS, resolution={resolution}")

    def animate_scenes(
        self,
        scene_script: dict,
        image_paths: list[str],
        output_dir: str
    ) -> list[str]:
        """
        Create animated clips for all scenes.

        Args:
            scene_script: Scene script JSON from scene_generator
            image_paths: List of image file paths (one per scene)
            output_dir: Directory to save video clips

        Returns:
            list: Paths to generated video clips

        Raises:
            ValueError: If inputs are invalid
        """
        if not scene_script or "scenes" not in scene_script:
            raise ValueError("Invalid scene_script: missing 'scenes'")

        scenes = scene_script["scenes"]

        if len(scenes) != len(image_paths):
            raise ValueError(
                f"Scene count mismatch: {len(scenes)} scenes vs {len(image_paths)} images"
            )

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        logger.info(f"Creating {len(scenes)} animated clips...")

        clip_paths = []

        for scene, image_path in zip(tqdm(scenes, desc="Animating scenes"), image_paths):
            # Validate image exists
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image not found: {image_path}")

            # Generate output filename
            scene_num = scene.get("scene_number", len(clip_paths) + 1)
            output_path = os.path.join(output_dir, f"scene_{scene_num:02d}.mp4")

            # Create the clip
            self._create_clip(image_path, scene, output_path)

            clip_paths.append(output_path)
            logger.debug(f"Created clip: {output_path}")

        logger.info(f"Generated {len(clip_paths)} clips")
        return clip_paths

    def _create_clip(
        self,
        image_path: str,
        scene: dict,
        output_path: str
    ) -> None:
        """
        Create a single animated clip from a static image.

        Args:
            image_path: Path to input image
            scene: Scene dict with duration and camera_movement
            output_path: Path to save output video

        Raises:
            ValueError: If scene parameters are invalid
        """
        # Get scene parameters
        duration = float(scene.get("duration", 3.0))
        movement = scene.get("camera_movement", "static")

        # Validate
        if movement not in self.VALID_MOVEMENTS:
            logger.warning(f"Invalid camera movement: {movement}. Using 'static'.")
            movement = "static"

        if duration <= 0:
            raise ValueError(f"Invalid duration: {duration}")

        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")

        # Resize to target resolution
        image = cv2.resize(image, self.resolution, interpolation=cv2.INTER_LANCZOS4)

        # Calculate number of frames
        total_frames = int(duration * self.fps)

        # Apply camera movement
        frames = self._apply_camera_movement(image, movement, total_frames)

        # Save as video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(
            output_path,
            fourcc,
            self.fps,
            self.resolution
        )

        for frame in frames:
            writer.write(frame)

        writer.release()

        # Verify file was created
        if not os.path.exists(output_path):
            raise RuntimeError(f"Video file not created: {output_path}")

        file_size = os.path.getsize(output_path)
        if file_size == 0:
            raise RuntimeError(f"Video file is empty: {output_path}")

        logger.debug(
            f"Clip created: {output_path} ({duration}s, {total_frames} frames, {file_size/1024:.1f} KB)"
        )

    def _apply_camera_movement(
        self,
        image: np.ndarray,
        movement: str,
        total_frames: int
    ) -> list[np.ndarray]:
        """
        Apply camera movement to an image.

        Args:
            image: Input image as numpy array
            movement: Type of camera movement
            total_frames: Number of frames to generate

        Returns:
            list: Frames with camera movement applied
        """
        height, width = image.shape[:2]
        frames = []

        if movement == "static":
            frames = self._apply_static(image, total_frames)

        elif movement == "zoom_in":
            frames = self._apply_zoom_in(image, total_frames)

        elif movement == "zoom_out":
            frames = self._apply_zoom_out(image, total_frames)

        elif movement == "pan_left":
            frames = self._apply_pan(image, total_frames, direction="left")

        elif movement == "pan_right":
            frames = self._apply_pan(image, total_frames, direction="right")

        else:
            logger.warning(f"Unknown movement: {movement}. Using static.")
            frames = self._apply_static(image, total_frames)

        return frames

    def _apply_static(
        self,
        image: np.ndarray,
        total_frames: int
    ) -> list[np.ndarray]:
        """Apply no movement - just duplicate the frame."""
        return [image.copy() for _ in range(total_frames)]

    def _apply_zoom_in(
        self,
        image: np.ndarray,
        total_frames: int
    ) -> list[np.ndarray]:
        """
        Apply gradual zoom in (100% to 120%).

        Uses smooth easing (ease-in-out) for natural motion.
        """
        height, width = image.shape[:2]
        frames = []

        # Zoom parameters
        start_scale = 1.0
        end_scale = 1.2

        for frame_idx in range(total_frames):
            # Calculate progress (0 to 1)
            t = frame_idx / (total_frames - 1) if total_frames > 1 else 0

            # Apply easing (ease-in-out)
            # t_eased = t < 0.5 ? 2 * t * t : 1 - pow(-2 * t + 2, 2) / 2
            if t < 0.5:
                t_eased = 2 * t * t
            else:
                t_eased = 1 - ((-2 * t + 2) ** 2) / 2

            # Calculate current scale
            scale = start_scale + (end_scale - start_scale) * t_eased

            # Calculate crop region (centered zoom)
            crop_width = int(width / scale)
            crop_height = int(height / scale)

            x1 = (width - crop_width) // 2
            y1 = (height - crop_height) // 2
            x2 = x1 + crop_width
            y2 = y1 + crop_height

            # Crop and resize
            cropped = image[y1:y2, x1:x2]
            frame = cv2.resize(cropped, (width, height), interpolation=cv2.INTER_LANCZOS4)
            frames.append(frame)

        return frames

    def _apply_zoom_out(
        self,
        image: np.ndarray,
        total_frames: int
    ) -> list[np.ndarray]:
        """
        Apply gradual zoom out (120% to 100%).

        Uses smooth easing (ease-in-out) for natural motion.
        """
        height, width = image.shape[:2]
        frames = []

        # Zoom parameters (reversed from zoom_in)
        start_scale = 1.2
        end_scale = 1.0

        for frame_idx in range(total_frames):
            # Calculate progress (0 to 1)
            t = frame_idx / (total_frames - 1) if total_frames > 1 else 0

            # Apply easing (ease-in-out)
            if t < 0.5:
                t_eased = 2 * t * t
            else:
                t_eased = 1 - ((-2 * t + 2) ** 2) / 2

            # Calculate current scale
            scale = start_scale + (end_scale - start_scale) * t_eased

            # Calculate crop region
            crop_width = int(width / scale)
            crop_height = int(height / scale)

            x1 = (width - crop_width) // 2
            y1 = (height - crop_height) // 2
            x2 = x1 + crop_width
            y2 = y1 + crop_height

            # Crop and resize
            cropped = image[y1:y2, x1:x2]
            frame = cv2.resize(cropped, (width, height), interpolation=cv2.INTER_LANCZOS4)
            frames.append(frame)

        return frames

    def _apply_pan(
        self,
        image: np.ndarray,
        total_frames: int,
        direction: str = "left"
    ) -> list[np.ndarray]:
        """
        Apply pan movement.

        Pans by 20% of image width.

        Args:
            image: Input image
            total_frames: Number of frames
            direction: "left" or "right"
        """
        height, width = image.shape[:2]
        frames = []

        # Pan amount (20% of width)
        pan_amount = int(width * 0.2)

        if direction == "left":
            # Start offset to right, pan to left
            start_offset = pan_amount
            end_offset = 0
        else:  # right
            # Start offset to left, pan to right
            start_offset = 0
            end_offset = pan_amount

        for frame_idx in range(total_frames):
            # Calculate progress with easing
            t = frame_idx / (total_frames - 1) if total_frames > 1 else 0

            # Ease-in-out
            if t < 0.5:
                t_eased = 2 * t * t
            else:
                t_eased = 1 - ((-2 * t + 2) ** 2) / 2

            # Calculate current offset
            offset = int(start_offset + (end_offset - start_offset) * t_eased)

            # Crop with offset
            x1 = offset
            x2 = x1 + width
            y1 = 0
            y2 = height

            # Handle boundaries
            if x2 > width:
                x2 = width
                x1 = width - width  # This shouldn't happen with proper bounds
            if x1 < 0:
                x1 = 0
                x2 = width

            cropped = image[y1:y2, x1:x2]
            frame = cv2.resize(cropped, (width, height), interpolation=cv2.INTER_LANCZOS4)
            frames.append(frame)

        return frames

    def get_clip_info(self, clip_path: str) -> dict:
        """
        Get information about a generated clip.

        Args:
            clip_path: Path to video clip

        Returns:
            dict: Clip information (duration, fps, frame_count, resolution, size)
        """
        cap = cv2.VideoCapture(clip_path)

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        cap.release()

        duration = frame_count / fps if fps > 0 else 0
        file_size = os.path.getsize(clip_path)

        return {
            "path": clip_path,
            "duration": duration,
            "fps": fps,
            "frame_count": frame_count,
            "resolution": (width, height),
            "file_size": file_size
        }
