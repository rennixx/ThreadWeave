"""
Image Generator Module

Generates scene images using Stable Diffusion XL Lightning or Z.AI API.
Optimized for speed with model caching and torch.compile.
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
    FluxPipeline,
    StableDiffusionXLPipeline,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler
)
from tqdm import tqdm
from huggingface_hub import snapshot_download
from transformers import CLIPTextModel, CLIPTokenizer

logger = logging.getLogger(__name__)

# Cache directory for models
MODELS_CACHE_DIR = os.path.expanduser("~/.cache/huggingface/hub")


class ImageGenerator:
    """
    Generates scene images using Stable Diffusion XL or Z.AI API.
    """

    def __init__(
        self,
        model_name: str = "black-forest-labs/FLUX.1-schnell",
        device: Optional[str] = None,
        use_api: bool = False,
        api_key: Optional[str] = None,
        api_provider: str = "zai",
        use_torch_compile: bool = True,
        cache_dir: str = None,
        model_type: str = "auto"
    ):
        """
        Initialize the image generator.

        Args:
            model_name: HuggingFace model name or local path (for local generation)
            device: "cuda", "cpu", or None to auto-detect
            use_api: Use API-based generation instead of local model
            api_key: API key for the service
            api_provider: "zai" for Z.AI image generation
            use_torch_compile: Enable torch.compile for faster inference (CUDA only)
            cache_dir: Custom cache directory for models
            model_type: "auto", "flux", "sdxl", or "sdt3"
        """
        self.model_name = model_name
        self.device = self._detect_device(device)
        self.pipeline = None
        self.use_api = use_api
        self.api_key = api_key
        self.api_provider = api_provider
        self.use_torch_compile = use_torch_compile and self.device == "cuda"
        self.cache_dir = cache_dir or os.path.join(os.getcwd(), "models", "cached")
        self.model_loaded = False
        self.model_type = self._detect_model_type(model_name, model_type)

        # Ensure cache directory exists
        os.makedirs(self.cache_dir, exist_ok=True)

        # Set HuggingFace cache environment
        os.environ['HUGGINGFACE_HUB_CACHE'] = self.cache_dir

        self._load_config()

        logger.info(f"ImageGenerator initialized with model: {model_name} (type: {self.model_type})")

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

    def _detect_model_type(self, model_name: str, model_type: str) -> str:
        """
        Detect the model type for pipeline selection.

        Args:
            model_name: Model name or path
            model_type: Explicit model type or "auto" to detect

        Returns:
            str: "flux", "sdxl", or "sdt3"
        """
        if model_type != "auto":
            return model_type.lower()

        model_name_lower = model_name.lower()

        if "flux" in model_name_lower:
            return "flux"
        elif "sdt3" in model_name_lower or "sd3" in model_name_lower:
            return "sdt3"
        else:
            # Default to SDXL for stability
            return "sdxl"

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

        # Default configs per model type
        default_configs = {
            "flux": {
                "image_steps": 4,
                "image_cfg_scale": 0.0,
                "image_resolution": [768, 1024],
                "guidance": 0.0,
                "max_sequence_length": 256
            },
            "sdxl": {
                "image_steps": 4,
                "image_cfg_scale": 0.0,
                "image_resolution": [768, 1024],
                "guidance": 0.0
            },
            "sdt3": {
                "image_steps": 28,
                "image_cfg_scale": 7.0,
                "image_resolution": [1024, 1024],
                "guidance": 7.0
            }
        }

        base_config = {
            "use_consistent_seed": True,
            "base_seed": 42,
            "use_torch_compile": True
        }

        # Get default for current model type
        default_config = {**base_config, **default_configs.get(self.model_type, default_configs["sdxl"])}

        if os.path.exists(config_path):
            import yaml
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                self.steps = config.get("image_steps", default_config["image_steps"])
                self.cfg_scale = config.get("image_cfg_scale", default_config["image_cfg_scale"])
                self.guidance = config.get("guidance", default_config.get("guidance", self.cfg_scale))
                self.resolution = tuple(config.get("image_resolution", default_config["image_resolution"]))
                self.use_consistent_seed = config.get("use_consistent_seed", default_config["use_consistent_seed"])
                self.base_seed = config.get("base_seed", default_config["base_seed"])
                self.max_sequence_length = config.get("max_sequence_length", default_config.get("max_sequence_length", 256))
        else:
            self.steps = default_config["image_steps"]
            self.cfg_scale = default_config["image_cfg_scale"]
            self.guidance = default_config.get("guidance", self.cfg_scale)
            self.resolution = tuple(default_config["image_resolution"])
            self.use_consistent_seed = default_config["use_consistent_seed"]
            self.base_seed = default_config["base_seed"]
            self.max_sequence_length = default_config.get("max_sequence_length", 256)

        if not self.use_api:
            logger.info(f"Config: model={self.model_type}, resolution={self.resolution}, steps={self.steps}, guidance={self.guidance}")

    def load_model(self) -> None:
        """
        Load the appropriate model pipeline (Flux, SDXL, or SD3) for local generation.

        Uses model caching to avoid re-downloading. Enables torch.compile for speed.
        """
        if self.use_api:
            logger.info("API mode enabled, skipping local model load")
            return

        if self.model_loaded:
            logger.info("Model already loaded, skipping...")
            return

        logger.info(f"Loading model: {self.model_name} (type: {self.model_type})")

        try:
            # Check if model is cached locally
            local_model_path = self._get_cached_model_path()

            model_path = local_model_path if local_model_path else self.model_name
            dtype = torch.float16 if self.device == "cuda" else torch.float32
            variant = "fp16" if self.device == "cuda" else None

            # Load the appropriate pipeline based on model type
            if self.model_type == "flux":
                self._load_flux_pipeline(model_path, dtype, variant)
            elif self.model_type == "sdt3":
                self._load_sdt3_pipeline(model_path, dtype, variant)
            else:  # sdxl
                self._load_sdxl_pipeline(model_path, dtype, variant)

            # Enable memory optimizations
            self._enable_memory_optimizations()

            self.model_loaded = True
            logger.info("Model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def _load_flux_pipeline(self, model_path: str, dtype: torch.dtype, variant: Optional[str]) -> None:
        """Load Flux.1 pipeline."""
        logger.info("Loading Flux.1 pipeline...")

        self.pipeline = FluxPipeline.from_pretrained(
            model_path,
            torch_dtype=dtype,
            use_safetensors=True,
            cache_dir=self.cache_dir,
            local_files_only=model_path != self.model_name
        )

        # Flux doesn't need scheduler changes
        self.pipeline = self.pipeline.to(self.device)

    def _load_sdxl_pipeline(self, model_path: str, dtype: torch.dtype, variant: Optional[str]) -> None:
        """Load SDXL Lightning pipeline."""
        logger.info("Loading SDXL Lightning pipeline...")

        self.pipeline = StableDiffusionXLPipeline.from_pretrained(
            model_path,
            torch_dtype=dtype,
            use_safetensors=True,
            variant=variant,
            cache_dir=self.cache_dir,
            local_files_only=model_path != self.model_name
        )

        # Set scheduler for Lightning
        self.pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
            self.pipeline.scheduler.config,
            timestep_spacing="trailing"
        )

        self.pipeline = self.pipeline.to(self.device)

    def _load_sdt3_pipeline(self, model_path: str, dtype: torch.dtype, variant: Optional[str]) -> None:
        """Load Stable Diffusion 3 pipeline."""
        logger.info("Loading Stable Diffusion 3 pipeline...")

        from diffusers import StableDiffusion3Pipeline

        self.pipeline = StableDiffusion3Pipeline.from_pretrained(
            model_path,
            torch_dtype=dtype,
            use_safetensors=True,
            variant=variant,
            cache_dir=self.cache_dir,
            local_files_only=model_path != self.model_name
        )

        self.pipeline = self.pipeline.to(self.device)

    def _enable_memory_optimizations(self) -> None:
        """Enable memory optimizations for CUDA."""
        if self.device == "cuda":
            # Enable xformers if available
            try:
                self.pipeline.enable_xformers_memory_efficient_attention()
                logger.info("xformers memory optimization enabled")
            except Exception:
                logger.info("xformers not available, using default attention")

            # Enable attention slicing for memory efficiency
            self.pipeline.enable_attention_slicing()

            # Enable VAE slicing to reduce memory
            if hasattr(self.pipeline, 'enable_vae_slicing'):
                self.pipeline.enable_vae_slicing()

            # Apply torch.compile for faster inference (2-3x speedup)
            if self.use_torch_compile:
                try:
                    logger.info("Applying torch.compile for faster inference...")
                    # Compile only the transformer/unet for best performance/memory tradeoff
                    if hasattr(self.pipeline, 'transformer'):
                        # Flux uses transformer
                        self.pipeline.transformer = torch.compile(
                            self.pipeline.transformer,
                            mode="reduce-overhead",
                            fullgraph=False
                        )
                    elif hasattr(self.pipeline, 'unet'):
                        # SDXL/SD3 use unet
                        self.pipeline.unet = torch.compile(
                            self.pipeline.unet,
                            mode="reduce-overhead",
                            fullgraph=False
                        )
                    logger.info("torch.compile applied successfully")
                except Exception as e:
                    logger.warning(f"torch.compile failed: {e}. Continuing without compile.")

    def _get_cached_model_path(self) -> Optional[str]:
        """
        Check if model is cached locally and return the path.

        Returns:
            Local path to cached model or None if not cached
        """
        # Try to find the model in the cache directory
        model_folder_name = self.model_name.replace("--", "/").replace("/", "--")
        cache_path = os.path.join(self.cache_dir, f"models--{model_folder_name}")

        if os.path.exists(cache_path):
            # Find the snapshot folder (usually named like 'snapshots/<hash>')
            snapshots_dir = os.path.join(cache_path, "snapshots")
            if os.path.exists(snapshots_dir):
                snapshots = [d for d in os.listdir(snapshots_dir) if os.path.isdir(os.path.join(snapshots_dir, d))]
                if snapshots:
                    # Return the most recent snapshot
                    latest_snapshot = sorted(snapshots)[-1]
                    return os.path.join(snapshots_dir, latest_snapshot)

        return None

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
        Generate an image using local model (Flux, SDXL, or SD3).

        Args:
            prompt: Text description for image generation
            negative_prompt: Things to avoid in the image (model-dependent)
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

        # Generate based on model type
        logger.info(f"Generating locally ({self.model_type}, {self.steps}-step): {prompt[:50]}...")

        if self.model_type == "flux":
            result = self._generate_flux(prompt, seed, generator)
        elif self.model_type == "sdt3":
            result = self._generate_sdt3(prompt, negative_prompt, seed, generator)
        else:  # sdxl
            result = self._generate_sdxl(prompt, seed, generator)

        return result.images[0] if hasattr(result, 'images') else result[0]

    def _generate_flux(self, prompt: str, seed: Optional[int], generator: Optional[torch.Generator]):
        """Generate with Flux pipeline."""
        return self.pipeline(
            prompt=prompt,
            num_inference_steps=self.steps,
            guidance=self.guidance,
            height=self.resolution[1],
            width=self.resolution[0],
            generator=generator
        )

    def _generate_sdxl(self, prompt: str, seed: Optional[int], generator: Optional[torch.Generator]):
        """Generate with SDXL pipeline."""
        return self.pipeline(
            prompt=prompt,
            num_inference_steps=self.steps,
            guidance_scale=self.guidance,
            height=self.resolution[1],
            width=self.resolution[0],
            generator=generator
        )

    def _generate_sdt3(self, prompt: str, negative_prompt: str, seed: Optional[int], generator: Optional[torch.Generator]):
        """Generate with SD3 pipeline."""
        return self.pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=self.steps,
            guidance_scale=self.guidance,
            height=self.resolution[1],
            width=self.resolution[0],
            generator=generator
        )

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
            self.model_loaded = False

            # Clear CUDA cache
            if self.device == "cuda":
                torch.cuda.empty_cache()

            logger.info("Model unloaded from memory")
