"""
AnimateDiff Generator Module

Generates true frame-by-frame animated videos from text prompts using AnimateDiff.
This creates actual motion and animation, not just zoom/pan on static images.
"""

import logging
import os
from typing import Optional, List
import torch
import numpy as np
from PIL import Image
from diffusers import (
    AnimateDiffPipeline,
    DDIMScheduler,
    MotionAdapter
)
from tqdm import tqdm

logger = logging.getLogger(__name__)


class AnimateDiffGenerator:
    """
    Generates animated videos using AnimateDiff for true frame-by-frame animation.

    Unlike simple image animation, this creates actual motion sequences from text prompts.
    Example: "A frog shooting its tongue to catch a fly" generates all frames showing
    the complete action sequence.
    """

    def __init__(
        self,
        model_name: str = "frankjoshua/toonyou_beta6",
        motion_adapter: str = "guoyww/animatediff-motion-adapter-v2",
        device: Optional[str] = None,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize the AnimateDiff generator.

        Args:
            model_name: Base SD model for image generation
            motion_adapter: Motion adapter for animation (v2 works without auth)
            device: "cuda", "cpu", or None to auto-detect
            cache_dir: Custom cache directory for models
        """
        self.device = self._detect_device(device)
        self.model_name = model_name
        self.motion_adapter_name = motion_adapter
        self.cache_dir = cache_dir or os.path.join(os.getcwd(), "models", "cached")
        self.pipeline = None
        self.model_loaded = False

        os.makedirs(self.cache_dir, exist_ok=True)
        os.environ['HUGGINGFACE_HUB_CACHE'] = self.cache_dir

        logger.info(f"AnimateDiffGenerator initialized: {model_name} + {motion_adapter}")

    def _detect_device(self, device: Optional[str]) -> str:
        """Auto-detect the best available device."""
        if device:
            return device
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def load_model(self) -> None:
        """Load the AnimateDiff pipeline with motion adapter."""
        if self.model_loaded:
            logger.info("Model already loaded")
            return

        logger.info("Loading AnimateDiff model (this may take a while on first run)...")

        try:
            # Load motion adapter
            logger.info(f"Loading motion adapter: {self.motion_adapter_name}")
            motion_adapter = MotionAdapter.from_pretrained(
                self.motion_adapter_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                cache_dir=self.cache_dir
            )

            # Load base pipeline
            logger.info(f"Loading base model: {self.model_name}")
            from diffusers import DPMSolverMultistepScheduler

            self.pipeline = AnimateDiffPipeline.from_pretrained(
                self.model_name,
                motion_adapter=motion_adapter,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                scheduler=DDIMScheduler.from_pretrained(
                    self.model_name,
                    subfolder="scheduler"
                ),
                cache_dir=self.cache_dir
            ).to(self.device)

            # Memory optimizations
            if self.device == "cuda":
                self.pipeline.enable_vae_slicing()
                try:
                    self.pipeline.enable_xformers_memory_efficient_attention()
                    logger.info("xformers enabled")
                except:
                    logger.info("xformers not available")

            self.model_loaded = True
            logger.info("AnimateDiff model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load AnimateDiff: {e}")
            raise

    def generate_animated_video(
        self,
        prompt: str,
        output_path: str,
        negative_prompt: str = "bad quality, worse quality, low resolution, blurry, distortion, ugly, deformed, cross-eyed",
        num_frames: int = 32,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 20,
        fps: int = 10,
        seed: Optional[int] = None
    ) -> str:
        """
        Generate an animated video from a text prompt.

        Args:
            prompt: Description of the animated scene
            output_path: Where to save the video
            negative_prompt: Things to avoid
            num_frames: Number of frames to generate (16-32 is typical)
            guidance_scale: CFG scale
            num_inference_steps: Denoising steps
            fps: Frames per second for output
            seed: Random seed for reproducibility

        Returns:
            str: Path to generated video
        """
        if not self.model_loaded:
            self.load_model()

        logger.info(f"Generating animated video: {prompt[:60]}...")

        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        # Generate the animation
        with torch.no_grad():
            result = self.pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_frames=num_frames,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                generator=generator,
                height=512,
                width=512
            )

        # Save frames as video
        frames = result.frames[0]
        self._save_frames_as_video(frames, output_path, fps)

        logger.info(f"Animation saved: {output_path}")
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

    def generate_from_scene_description(
        self,
        scene_description: str,
        output_path: str,
        duration: float = 3.0,
        fps: int = 10,
        seed: Optional[int] = None
    ) -> str:
        """
        Generate animation from a simple scene description.

        Args:
            scene_description: What to animate
            output_path: Output video path
            duration: Duration in seconds
            fps: Frames per second
            seed: Random seed

        Returns:
            str: Path to generated video
        """
        num_frames = int(duration * fps)

        # Build a detailed prompt for animation
        detailed_prompt = self._build_animation_prompt(scene_description)

        return self.generate_animated_video(
            prompt=detailed_prompt,
            output_path=output_path,
            num_frames=num_frames,
            fps=fps,
            seed=seed
        )

    def _build_animation_prompt(self, description: str) -> str:
        """Build a detailed prompt optimized for AnimateDiff."""
        # AnimateDiff works best with detailed, motion-oriented prompts
        return (
            f"{description}, "
            f"high quality, detailed, smooth animation, "
            f"consistent character design, fluid motion, "
            f"cinematic lighting, professional animation"
        )

    def unload_model(self) -> None:
        """Unload model from memory."""
        if self.pipeline:
            del self.pipeline
            self.pipeline = None
            self.model_loaded = False

            if self.device == "cuda":
                torch.cuda.empty_cache()

            logger.info("Model unloaded")
