"""
Image Generator Module

Generates scene images using Stable Diffusion XL or Z.AI API.
"""

import json
import logging
import os
from typing import Optional
from pathlib import Path

import torch
import requests
from PIL import Image
from diffusers import (
    StableDiffusionXLPipeline,
    DPMSolverMultistepScheduler
)
from tqdm import tqdm

logger = logging.getLogger(__name__)


class ImageGenerator:
    """
    Generates scene images using Stable Diffusion XL or Z.AI API.
    """

    def __init__(
        self,
        model_name: str = "stabilityai/sdxl-turbo",
        device: Optional[str] = None,
        use_api: bool = False,
        api_key: Optional[str] = None,
        api_provider: str = "zai"
    ):
        """
        Initialize the image generator.

        Args:
            model_name: HuggingFace model name or local path (for local generation)
            device: "cuda", "cpu", or None to auto-detect
            use_api: Use API-based generation instead of local model
            api_key: API key for the service
            api_provider: "zai" for Z.AI image generation
        """
        self.model_name = model_name
        self.device = self._detect_device(device)
        self.pipeline = None
        self.use_api = use_api
        self.api_key = api_key
        self.api_provider = api_provider

        self._load_config()

        if not use_api:
            # Check CUDA availability
            if self.device == "cuda":
                if not torch.cuda.is_available():
                    logger.warning("CUDA requested but not available. Falling back to CPU.")
                    self.device = "cpu"
                else:
                    gpu_name = torch.cuda.get_device_name(0)
                    logger.info(f"Using GPU: {gpu_name}")
        else:
            logger.info(f"Using {api_provider.upper()} API for image generation")

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

        if not self.use_api:
            logger.info(f"Config: resolution={self.resolution}, steps={self.steps}, cfg={self.cfg_scale}")

    def load_model(self) -> None:
        """
        Load the Stable Diffusion model (for local generation only).

        This is done separately to allow control over when the model is loaded.
        """
        if self.use_api:
            logger.info("API mode enabled, skipping local model load")
            return

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

            logger.info("Model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def _generate_via_api(self, prompt: str, negative_prompt: str = "") -> Image.Image:
        """
        Generate an image using Z.AI API.

        Args:
            prompt: Text description for image generation
            negative_prompt: Things to avoid in the image

        Returns:
            PIL Image
        """
        if not self.api_key:
            raise ValueError(f"API key required for {self.api_provider} image generation")

        if self.api_provider == "zai":
            return self._generate_via_zai(prompt, negative_prompt)
        else:
            raise ValueError(f"Unsupported API provider: {self.api_provider}")

    def _generate_via_zai(self, prompt: str, negative_prompt: str) -> Image.Image:
        """
        Generate image using Z.AI's cogview-3 model.

        Z.AI API documentation: https://open.bigmodel.cn/doc/api#images
        """
        url = "https://open.bigmodel.cn/api/paas/v4/images/generations"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": "cogview-3",
            "prompt": prompt,
            "size": f"{self.resolution[0]}x{self.resolution[1]}",
            "n": 1
        }

        if negative_prompt:
            # Z.AI might not support negative_prompt directly, include it in the main prompt
            payload["prompt"] = f"{prompt}. Avoid: {negative_prompt}"

        try:
            logger.info(f"Sending request to Z.AI API...")
            response = requests.post(url, headers=headers, json=payload, timeout=60)

            if response.status_code != 200:
                error_msg = response.text
                logger.error(f"Z.AI API error: {error_msg}")
                raise RuntimeError(f"Z.AI API returned {response.status_code}: {error_msg}")

            data = response.json()

            # Extract image URL from response
            if "data" in data and len(data["data"]) > 0:
                image_url = data["data"][0]["url"]

                # Download the image
                logger.info(f"Downloading image from {image_url}...")
                img_response = requests.get(image_url, timeout=30)
                img_response.raise_for_status()

                from io import BytesIO
                image = Image.open(BytesIO(img_response.content))
                return image
            else:
                raise RuntimeError("No image data in Z.AI response")

        except Exception as e:
            logger.error(f"Z.AI image generation failed: {e}")
            raise

    def _generate_via_local(self, prompt: str, negative_prompt: str = "", seed: Optional[int] = None) -> Image.Image:
        """
        Generate an image using local Stable Diffusion model.

        Args:
            prompt: Text description for image generation
            negative_prompt: Things to avoid in the image
            seed: Random seed for reproducibility

        Returns:
            PIL Image
        """
        if not self.pipeline:
            self.load_model()

        # Set seed for reproducibility
        if seed is None and self.use_consistent_seed:
            seed = self.base_seed

        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        # Generate
        logger.info(f"Generating locally: {prompt[:50]}...")
        result = self.pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=self.steps,
            guidance_scale=self.cfg_scale,
            height=self.resolution[1],
            width=self.resolution[0],
            generator=generator
        )

        return result.images[0]

    def generate_scene_images(
        self,
        scene_script: dict,
        output_dir: str
    ) -> list[str]:
        """
        Generate images for all scenes in the script.

        Args:
            scene_script: Scene script from SceneGenerator
            output_dir: Directory to save generated images

        Returns:
            list: Paths to generated images
        """
        # Load prompt templates
        config_path = "config/prompt_templates.json"
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                templates = json.load(f)
                positive_template = templates.get("image_generation", {}).get(
                    "positive_prompt_template",
                    "{visual_description}, {art_style}, high quality"
                )
                negative_template = templates.get("image_generation", {}).get(
                    "negative_prompt",
                    "blurry, distorted, low quality"
                )
        else:
            positive_template = "{visual_description}, {art_style}, high quality"
            negative_template = "blurry, distorted, low quality"

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        scenes = scene_script.get("scenes", [])
        metadata = scene_script.get("metadata", {})
        art_style = metadata.get("art_style", "")

        image_paths = []
        total_scenes = len(scenes)

        logger.info(f"Generating {total_scenes} scene images...")

        for scene in tqdm(scenes, desc="Generating images"):
            scene_num = scene.get("scene_number", 1)
            visual_desc = scene.get("visual_description", "")
            camera = scene.get("camera_movement", "static")

            # Build prompts
            positive_prompt = positive_template.format(
                visual_description=visual_desc,
                art_style=art_style
            )

            # Add camera movement hints to prompt
            if "zoom_in" in camera:
                positive_prompt += ", close-up shot, detailed"
            elif "zoom_out" in camera:
                positive_prompt += ", wide angle shot, expansive"
            elif "pan" in camera:
                positive_prompt += ", panoramic view, horizontal composition"

            negative_prompt = negative_template

            # Generate image
            if self.use_api:
                image = self._generate_via_api(positive_prompt, negative_prompt)
            else:
                seed = self.base_seed + scene_num if self.use_consistent_seed else None
                image = self._generate_via_local(positive_prompt, negative_prompt, seed)

            # Save image
            filename = f"scene_{scene_num:03d}.png"
            filepath = os.path.join(output_dir, filename)
            image.save(filepath)
            image_paths.append(filepath)

            logger.debug(f"Saved: {filepath}")

        logger.info(f"Generated {len(image_paths)} images")
        return image_paths

    def unload_model(self) -> None:
        """
        Unload the model from memory to free up resources.
        """
        if self.pipeline:
            del self.pipeline
            self.pipeline = None

            # Clear CUDA cache
            if self.device == "cuda":
                torch.cuda.empty_cache()

            logger.info("Model unloaded from memory")
