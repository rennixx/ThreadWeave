"""
Video Generation Module

Generates smooth frame-by-frame animations using state-of-the-art models:
- AnimateDiff v3 for character/motion animation
- Stable Video Diffusion (SVD) for image-to-video
- ModelScope for high-quality video generation
"""

import logging
import os
from typing import Optional, List
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from diffusers import (
    AnimateDiffPipeline,
    DPMSolverMultistepScheduler,
    StableVideoDiffusionPipeline
)
from tqdm import tqdm

logger = logging.getLogger(__name__)


class VideoGenerator:
    """
    Generates smooth video clips from images using state-of-the-art AI models.

    Supports:
    - AnimateDiff v3 for frame-by-frame animation with motion
    - Stable Video Diffusion (SVD) for image-to-video conversion
    """

    def __init__(
        self,
        model_type: str = "animatediff",
        device: Optional[str] = None,
        cache_dir: str = None,
        use_torch_compile: bool = True
    ):
        """
        Initialize the video generator.

        Args:
            model_type: "animatediff" or "svd"
            device: "cuda", "cpu", or None to auto-detect
            cache_dir: Custom cache directory for models
            use_torch_compile: Enable torch.compile for faster inference
        """
        self.model_type = model_type.lower()
        self.device = self._detect_device(device)
        self.cache_dir = cache_dir or os.path.join(os.getcwd(), "models", "cached")
        self.use_torch_compile = use_torch_compile and self.device == "cuda"
        self.pipeline = None
        self.model_loaded = False

        # Ensure cache directory exists
        os.makedirs(self.cache_dir, exist_ok=True)
        os.environ['HUGGINGFACE_HUB_CACHE'] = self.cache_dir

        logger.info(f"VideoGenerator initialized: {model_type} on {self.device}")

    def _detect_device(self, device: Optional[str]) -> str:
        """Auto-detect the best available device."""
        if device:
            return device
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def load_model(self) -> None:
        """Load the appropriate video generation model."""
        if self.model_loaded:
            logger.info("Model already loaded, skipping...")
            return

        if self.model_type == "animatediff":
            self._load_animatediff()
        elif self.model_type == "svd":
            self._load_svd()
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        self._enable_optimizations()
        self.model_loaded = True
        logger.info(f"{self.model_type.upper()} model loaded successfully")

    def _load_animatediff(self) -> None:
        """Load AnimateDiff v3 pipeline."""
        logger.info("Loading AnimateDiff v3 pipeline...")

        # Use the popular AnimateDiff v3 model
        model_id = "HotshotCo/animatediff-motion-module-v3"

        # Load SDXL base with AnimateDiff motion module
        from diffusers import AnimateDiffSDXLPipeline

        self.pipeline = AnimateDiffSDXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            motion_module=model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            cache_dir=self.cache_dir
        ).to(self.device)

        # Set scheduler
        self.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipeline.scheduler.config
        )

    def _load_svd(self) -> None:
        """Load Stable Video Diffusion pipeline."""
        logger.info("Loading Stable Video Diffusion pipeline...")

        self.pipeline = StableVideoDiffusionPipeline.from_pretrained(
            "stabilityai/stable-video-diffusion-img2vid-xt",
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            cache_dir=self.cache_dir
        ).to(self.device)

    def _enable_optimizations(self) -> None:
        """Enable memory and performance optimizations."""
        if self.device == "cuda":
            # Enable xformers if available
            try:
                self.pipeline.enable_xformers_memory_efficient_attention()
                logger.info("xformers memory optimization enabled")
            except Exception:
                logger.info("xformers not available, using default attention")

            # Enable attention slicing
            self.pipeline.enable_attention_slicing()

            # Enable VAE slicing for SVD
            if hasattr(self.pipeline, 'enable_vae_slicing'):
                self.pipeline.enable_vae_slicing()

            # Apply torch.compile
            if self.use_torch_compile:
                try:
                    logger.info("Applying torch.compile...")
                    if hasattr(self.pipeline, 'unet'):
                        self.pipeline.unet = torch.compile(
                            self.pipeline.unet,
                            mode="reduce-overhead",
                            fullgraph=False
                        )
                    logger.info("torch.compile applied")
                except Exception as e:
                    logger.warning(f"torch.compile failed: {e}")

    def generate_video_from_image(
        self,
        image_path: str,
        output_path: str,
        duration: float = 3.0,
        fps: int = 30,
        motion_bucket_id: int = 127,
        cond_aug: float = 0.02
    ) -> str:
        """
        Generate a smooth video clip from a static image using SVD.

        Args:
            image_path: Path to input image
            output_path: Path to save output video
            duration: Duration in seconds
            fps: Frames per second
            motion_bucket_id: How much motion to generate (1-255)
            cond_aug: Conditioning augmentation for variety

        Returns:
            str: Path to generated video
        """
        if not self.model_loaded:
            self.load_model()

        if self.model_type != "svd":
            logger.warning("generate_video_from_image works best with SVD model")

        logger.info(f"Generating video from {image_path}...")

        # Load and prepare image
        image = Image.open(image_path).convert("RGB")
        image = image.resize((1024, 576), Image.Resampling.LANCZOS)

        # Generate video frames
        num_frames = int(duration * fps)

        with torch.no_grad():
            result = self.pipeline(
                image=image,
                num_frames=num_frames,
                num_inference_steps=25,
                min_guidance_scale=1.0,
                max_guidance_scale=3.0,
                motion_bucket_id=motion_bucket_id,
                cond_aug=cond_aug,
                generator=torch.Generator(device=self.device).manual_seed(42)
            )

        # Save video
        frames = result.frames[0]
        self._save_frames_as_video(frames, output_path, fps)

        logger.info(f"Video saved to {output_path}")
        return output_path

    def generate_animated_clip(
        self,
        prompt: str,
        output_path: str,
        duration: float = 3.0,
        fps: int = 30,
        negative_prompt: str = "bad quality, worse quality, low resolution",
        num_inference_steps: int = 20,
        guidance_scale: float = 7.5
    ) -> str:
        """
        Generate an animated clip from text using AnimateDiff.

        Args:
            prompt: Text description of the animation
            output_path: Path to save output video
            duration: Duration in seconds
            fps: Frames per second
            negative_prompt: Things to avoid
            num_inference_steps: Number of denoising steps
            guidance_scale: CFG scale

        Returns:
            str: Path to generated video
        """
        if not self.model_loaded:
            self.load_model()

        if self.model_type != "animatediff":
            logger.warning("generate_animated_clip requires AnimateDiff model")

        logger.info(f"Generating animated clip: {prompt[:50]}...")

        num_frames = int(duration * fps)

        with torch.no_grad():
            result = self.pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_frames=num_frames,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=torch.Generator(device=self.device).manual_seed(42)
            )

        # Save video
        frames = result.frames[0]
        self._save_frames_as_video(frames, output_path, fps)

        logger.info(f"Animation saved to {output_path}")
        return output_path

    def _save_frames_as_video(self, frames: torch.Tensor, output_path: str, fps: int) -> None:
        """Save frames as MP4 video using OpenCV."""
        import cv2

        # Convert to numpy and denormalize
        frames = (frames * 255).cpu().numpy().astype(np.uint8)

        # Get dimensions
        num_frames, height, width, _ = frames.shape

        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # Write frames
        for frame in frames:
            # Convert RGB to BGR
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            writer.write(frame_bgr)

        writer.release()

    def generate_video_from_images(
        self,
        image_paths: List[str],
        output_path: str,
        duration_per_image: float = 2.0,
        fps: int = 30,
        use_motion: bool = True
    ) -> str:
        """
        Generate a smooth video by animating between multiple images.

        Args:
            image_paths: List of image paths
            output_path: Path to save output video
            duration_per_image: Duration for each image section
            fps: Frames per second
            use_motion: Use SVD for motion instead of simple fade

        Returns:
            str: Path to generated video
        """
        if not self.model_loaded:
            self.load_model()

        logger.info(f"Generating video from {len(image_paths)} images...")

        import cv2

        all_frames = []
        frames_per_image = int(duration_per_image * fps)

        for i, image_path in enumerate(tqdm(image_paths, desc="Processing images")):
            if use_motion and self.model_type == "svd":
                # Generate motion video for this image
                temp_output = f"{output_path}.temp_{i}.mp4"
                self.generate_video_from_image(
                    image_path,
                    temp_output,
                    duration=duration_per_image,
                    fps=fps
                )

                # Extract frames
                cap = cv2.VideoCapture(temp_output)
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    all_frames.append(frame)
                cap.release()

                # Clean up temp file
                if os.path.exists(temp_output):
                    os.remove(temp_output)
            else:
                # Simple frame duplication with crossfade
                image = cv2.imread(image_path)
                image = cv2.resize(image, (1024, 576))

                for _ in range(frames_per_image):
                    all_frames.append(image.copy())

        # Save final video
        if all_frames:
            height, width = all_frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            for frame in all_frames:
                writer.write(frame)

            writer.release()

        logger.info(f"Video saved to {output_path}")
        return output_path

    def unload_model(self) -> None:
        """Unload the model from memory."""
        if self.pipeline:
            del self.pipeline
            self.pipeline = None
            self.model_loaded = False

            if self.device == "cuda":
                torch.cuda.empty_cache()

            logger.info("Model unloaded from memory")
