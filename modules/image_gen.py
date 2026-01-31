"""
Image Generator Module

Generates consistent scene images using Stable Diffusion XL.
"""

import json
import logging
import os
from typing import Optional

import torch
from PIL import Image
from diffusers import (
    StableDiffusionXLPipeline,
    DPMSolverMultistepScheduler
)
from tqdm import tqdm

logger = logging.getLogger(__name__)


class ImageGenerator:
    """
    Generates scene images using Stable Diffusion XL.

    Optimized for RTX 4070 with GPU acceleration and memory management.
    """

    def __init__(
        self,
        model_name: str = "stabilityai/sdxl-turbo",
        device: Optional[str] = None
    ):
        """
        Initialize the image generator.

        Args:
            model_name: HuggingFace model name or local path
            device: "cuda", "cpu", or None to auto-detect
        """
        self.model_name = model_name
        self.device = self._detect_device(device)
        self.pipeline = None
        self._load_config()

        # Check CUDA availability
        if self.device == "cuda":
            if not torch.cuda.is_available():
                logger.warning("CUDA requested but not available. Falling back to CPU.")
                self.device = "cpu"
            else:
                gpu_name = torch.cuda.get_device_name(0)
                logger.info(f"Using GPU: {gpu_name}")

    def _detect_device(self, device: Optional[str]) -> str:
        """Auto-detect the best available device."""
        if device:
            return device

        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def _load_config(self) -> None:
        """Load configuration from settings file."""
        config_path = "config/settings.yaml"

        default_config = {
            "image_steps": 6,
            "image_cfg_scale": 2.0,
            "image_resolution": [768, 1024],
            "use_consistent_seed": True,
            "base_seed": 42
        }

        if os.path.exists(config_path):
            import yaml
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                self.steps = config.get("image_steps", default_config["image_steps"])
                self.cfg_scale = config.get("image_cfg_scale", default_config["image_cfg_scale"])
                self.resolution = tuple(config.get("image_resolution", default_config["image_resolution"]))
                self.use_consistent_seed = config.get("use_consistent_seed", default_config["use_consistent_seed"])
                self.base_seed = config.get("base_seed", default_config["base_seed"])
        else:
            self.steps = default_config["image_steps"]
            self.cfg_scale = default_config["image_cfg_scale"]
            self.resolution = tuple(default_config["image_resolution"])
            self.use_consistent_seed = default_config["use_consistent_seed"]
            self.base_seed = default_config["base_seed"]

        logger.info(f"Config: resolution={self.resolution}, steps={self.steps}, cfg={self.cfg_scale}")

    def load_model(self) -> None:
        """
        Load the Stable Diffusion model.

        This is done separately to allow control over when the model is loaded.
        """
        logger.info(f"Loading model: {self.model_name}")

        try:
            # Load the pipeline
            self.pipeline = StableDiffusionXLPipeline.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                use_safetensors=True,
                variant="fp16" if self.device == "cuda" else None
            )

            # Set scheduler
            self.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
                self.pipeline.scheduler.config
            )

            # Move to device
            self.pipeline = self.pipeline.to(self.device)

            # Enable memory optimizations
            if self.device == "cuda":
                # Enable xformers if available
                try:
                    self.pipeline.enable_xformers_memory_efficient_attention()
                    logger.info("xformers memory optimization enabled")
                except Exception:
                    logger.info("xformers not available, using default attention")

                # Enable attention slicing for memory efficiency
                self.pipeline.enable_attention_slicing()

                # Enable VAE slicing for large images
                self.pipeline.enable_vae_slicing()

            logger.info("Model loaded successfully")

        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")

    def unload_model(self) -> None:
        """Unload the model to free GPU memory."""
        if self.pipeline:
            del self.pipeline
            self.pipeline = None

        if self.device == "cuda":
            torch.cuda.empty_cache()
            logger.info("Model unloaded, GPU cache cleared")

    def generate_scene_images(
        self,
        scene_script: dict,
        output_dir: str
    ) -> list[str]:
        """
        Generate images for all scenes in the script.

        Args:
            scene_script: Scene script JSON from scene_generator
            output_dir: Directory to save images

        Returns:
            list: Paths to generated images

        Raises:
            ValueError: If scene_script is invalid
            RuntimeError: If generation fails
        """
        if not scene_script:
            raise ValueError("scene_script cannot be empty")

        if "scenes" not in scene_script:
            raise ValueError("scene_script must contain 'scenes' key")

        if "metadata" not in scene_script:
            raise ValueError("scene_script must contain 'metadata' key")

        scenes = scene_script["scenes"]
        art_style = scene_script["metadata"].get("art_style", "")

        if not scenes:
            raise ValueError("No scenes to generate")

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Load model if not loaded
        if self.pipeline is None:
            self.load_model()

        logger.info(f"Generating {len(scenes)} scene images...")

        image_paths = []
        seed = self.base_seed if self.use_consistent_seed else None

        for scene in tqdm(scenes, desc="Generating images"):
            try:
                # Generate image
                image = self._generate_single_image(
                    scene,
                    art_style,
                    seed
                )

                # Save image
                filename = f"scene_{scene['scene_number']:02d}.png"
                output_path = os.path.join(output_dir, filename)

                image.save(output_path, format="PNG", quality=95)
                image_paths.append(output_path)

                logger.debug(f"Saved: {output_path}")

            except Exception as e:
                logger.error(f"Failed to generate scene {scene.get('scene_number', '?')}: {e}")
                raise

        logger.info(f"Generated {len(image_paths)} images")
        return image_paths

    def _generate_single_image(
        self,
        scene: dict,
        art_style: str,
        seed: Optional[int] = None
    ) -> Image.Image:
        """
        Generate a single image from a scene description.

        Args:
            scene: Scene dict with visual_description
            art_style: Art style string
            seed: Random seed for consistency

        Returns:
            PIL.Image: Generated image
        """
        visual_description = scene.get("visual_description", "")

        if not visual_description:
            raise ValueError("Scene must have 'visual_description'")

        # Build prompts
        positive_prompt, negative_prompt = self._build_prompt(
            visual_description,
            art_style
        )

        # Set seed for consistency
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        # Generate
        width, height = self.resolution

        try:
            result = self.pipeline(
                prompt=positive_prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_inference_steps=self.steps,
                guidance_scale=self.cfg_scale,
                generator=generator,
                num_images=1
            )

            image = result.images[0]
            return image

        except torch.cuda.OutOfMemoryError:
            logger.error("GPU out of memory. Try reducing resolution or batch size.")
            self.unload_model()
            raise RuntimeError("GPU out of memory")

    def _build_prompt(
        self,
        visual_description: str,
        art_style: str
    ) -> tuple[str, str]:
        """
        Build positive and negative prompts for image generation.

        Args:
            visual_description: Scene visual description
            art_style: Art style keywords

        Returns:
            tuple: (positive_prompt, negative_prompt)
        """
        # Load templates from config if available
        config_path = "config/prompt_templates.json"

        default_positive_template = "{visual_description}, {art_style}, consistent style, professional illustration, clean composition, cinematic lighting, high quality, sharp focus, detailed, aesthetic"
        default_negative = "blurry, distorted, inconsistent style, multiple styles, text, watermark, signature, low quality, artifacts, deformed, ugly, bad anatomy, duplicate, cropped, worst quality, jpeg artifacts"

        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                templates = json.load(f)
                positive_template = templates.get("image_generation", {}).get(
                    "positive_prompt_template",
                    default_positive_template
                )
                negative_prompt = templates.get("image_generation", {}).get(
                    "negative_prompt",
                    default_negative
                )
        else:
            positive_template = default_positive_template
            negative_prompt = default_negative

        # Build positive prompt
        positive_prompt = positive_template.format(
            visual_description=visual_description,
            art_style=art_style
        )

        return positive_prompt, negative_prompt

    def get_memory_info(self) -> dict:
        """
        Get GPU memory usage information.

        Returns:
            dict: Memory info with keys: allocated, reserved, free, total
        """
        if self.device != "cuda":
            return {"device": "cpu"}

        return {
            "device": "cuda",
            "allocated": torch.cuda.memory_allocated() / 1024**3,  # GB
            "reserved": torch.cuda.memory_reserved() / 1024**3,  # GB
            "free": (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1024**3,  # GB
            "total": torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
        }
