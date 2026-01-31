"""
Scene Generator Module

Uses LLM to convert thread text into structured video scene scripts.
"""

import json
import logging
import os
import re
from typing import Optional

import yaml
from openai import OpenAI
from anthropic import Anthropic
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Valid camera movements
VALID_CAMERA_MOVEMENTS = ["static", "zoom_in", "zoom_out", "pan_left", "pan_right"]


class SceneGenerator:
    """
    Generates video scene scripts from Twitter thread content using LLMs.

    Supports OpenAI GPT and Anthropic Claude APIs.
    """

    def __init__(self, api_key: str, provider: str = "openai"):
        """
        Initialize the scene generator.

        Args:
            api_key: API key for the LLM provider
            provider: "openai" or "anthropic"
        """
        self.api_key = api_key
        self.provider = provider.lower()
        self.client = None
        self.model = None

        # Load prompt template from config
        self._load_prompt_template()

        # Initialize client
        if self.provider == "openai":
            self.client = OpenAI(api_key=api_key)
            self.model = "gpt-4o"  # or "gpt-4-turbo"
            logger.info(f"SceneGenerator initialized with OpenAI ({self.model})")
        elif self.provider == "anthropic":
            self.client = Anthropic(api_key=api_key)
            self.model = "claude-3-5-sonnet-20241022"  # Claude 3.5 Sonnet
            logger.info(f"SceneGenerator initialized with Anthropic ({self.model})")
        else:
            raise ValueError(f"Unsupported provider: {provider}. Use 'openai' or 'anthropic'.")

    def _load_prompt_template(self) -> None:
        """Load prompt template from config file."""
        config_path = "config/prompt_templates.json"

        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                templates = json.load(f)
                self.prompt_template = templates.get("scene_generation", {}).get(
                    "user_prompt_template", ""
                )
                self.system_prompt = templates.get("scene_generation", {}).get(
                    "system_prompt", ""
                )
                self.retry_prompt = templates.get("scene_generation", {}).get(
                    "retry_prompt", ""
                )
        else:
            # Fallback template
            self.system_prompt = (
                "You are a creative director converting social media threads "
                "into animated video scripts."
            )
            self.prompt_template = self._get_fallback_template()
            self.retry_prompt = "Please provide ONLY valid JSON."

    def _get_fallback_template(self) -> str:
        """Get fallback prompt template if config not found."""
        return (
            "Convert the following Twitter thread into a {target_duration}-second "
            "animated video script.\n\nTHREAD CONTENT:\n{thread_text}\n\n"
            "ART STYLE: {art_style}\n\n"
            "Create a JSON response with this exact structure:\n"
            "{{\n  \"metadata\": {{\n    \"art_style\": \"{art_style}\",\n"
            "    \"total_duration\": {target_duration},\n    \"scene_count\": <number>\n  }},\n"
            "  \"scenes\": [\n    {{\n      \"scene_number\": 1,\n      \"duration\": 3.5,\n"
            "      \"visual_description\": \"detailed description\",\n"
            "      \"narration_text\": \"narration\",\n"
            "      \"camera_movement\": \"static | zoom_in | zoom_out | pan_left | pan_right\"\n    }}\n  ]\n"
            "}}\n\n"
            "RULES:\n"
            "1. Create 6-10 scenes total\n"
            "2. Each scene should be 2-5 seconds\n"
            "3. Total duration should equal {target_duration} seconds (±3 seconds acceptable)\n"
            "4. Visual descriptions MUST use consistent style keywords\n"
            "5. Narration should capture the thread's key insights\n"
            "Respond with ONLY the JSON, no additional text."
        )

    def generate_scenes(
        self,
        thread_data: dict,
        art_style: str,
        target_duration: int = 30
    ) -> dict:
        """
        Generate scene script from thread data.

        Args:
            thread_data: Thread data from scraper (must contain 'combined_text')
            art_style: Art style description for visuals
            target_duration: Target video duration in seconds

        Returns:
            dict: Scene script with metadata and scenes array

        Raises:
            ValueError: If thread_data is invalid
            RuntimeError: If generation fails
        """
        # Validate input
        if "combined_text" not in thread_data:
            raise ValueError("thread_data must contain 'combined_text'")

        thread_text = thread_data["combined_text"]

        if not thread_text or len(thread_text.strip()) < 50:
            raise ValueError("Thread text is too short to generate scenes")

        logger.info(f"Generating scenes for thread {thread_data.get('thread_id', 'unknown')}")
        logger.info(f"Target duration: {target_duration}s, Style: {art_style}")

        # Build prompt
        prompt = self._build_prompt(thread_text, art_style, target_duration)

        # Generate with retry logic
        max_retries = 3

        for attempt in range(max_retries):
            try:
                response_text = self._call_llm(prompt, attempt)

                # Parse JSON response
                scene_data = self._parse_response(response_text)

                # Validate the response
                self._validate_scene_data(scene_data, target_duration)

                logger.info(
                    f"Successfully generated {scene_data['metadata']['scene_count']} scenes"
                )

                return scene_data

            except json.JSONDecodeError as e:
                logger.warning(f"Attempt {attempt + 1}: Invalid JSON - {e}")

                if attempt < max_retries - 1:
                    # Retry with retry prompt
                    logger.info("Retrying with retry prompt...")
                    prompt = self.retry_prompt
                else:
                    raise RuntimeError(
                        f"Failed to generate valid JSON after {max_retries} attempts"
                    )

            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying...")
                else:
                    raise RuntimeError(f"Scene generation failed: {e}")

        raise RuntimeError("Scene generation failed")

    def _call_llm(self, prompt: str, attempt: int) -> str:
        """
        Call the LLM API.

        Args:
            prompt: The prompt to send
            attempt: Current attempt number (for logging)

        Returns:
            str: LLM response text
        """
        if self.provider == "openai":
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=4000
            )
            return response.choices[0].message.content

        elif self.provider == "anthropic":
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4000,
                temperature=0.7,
                system=self.system_prompt,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return response.content[0].text

        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def _build_prompt(
        self,
        thread_text: str,
        art_style: str,
        target_duration: int
    ) -> str:
        """
        Build the LLM prompt.

        Args:
            thread_text: Combined thread text
            art_style: Art style description
            target_duration: Target duration in seconds

        Returns:
            str: Complete prompt
        """
        return self.prompt_template.format(
            thread_text=thread_text,
            art_style=art_style,
            target_duration=target_duration
        )

    def _parse_response(self, response_text: str) -> dict:
        """
        Parse JSON from LLM response.

        Handles common issues like markdown code blocks.

        Args:
            response_text: Raw LLM response

        Returns:
            dict: Parsed scene data

        Raises:
            json.JSONDecodeError: If JSON is invalid
        """
        # Remove markdown code blocks if present
        response_text = response_text.strip()

        # Look for JSON in code blocks
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
        if json_match:
            response_text = json_match.group(1)

        # Try to find JSON object boundaries
        if not response_text.startswith('{'):
            start_idx = response_text.find('{')
            if start_idx != -1:
                response_text = response_text[start_idx:]

        if not response_text.endswith('}'):
            end_idx = response_text.rfind('}')
            if end_idx != -1:
                response_text = response_text[:end_idx + 1]

        return json.loads(response_text)

    def _validate_scene_data(self, scene_data: dict, target_duration: int) -> None:
        """
        Validate scene data structure and content.

        Args:
            scene_data: Scene data to validate
            target_duration: Expected target duration

        Raises:
            ValueError: If validation fails
        """
        # Check structure
        if "metadata" not in scene_data:
            raise ValueError("Missing 'metadata' in response")

        if "scenes" not in scene_data:
            raise ValueError("Missing 'scenes' in response")

        metadata = scene_data["metadata"]
        scenes = scene_data["scenes"]

        # Validate metadata
        required_metadata_keys = ["art_style", "total_duration", "scene_count"]
        for key in required_metadata_keys:
            if key not in metadata:
                raise ValueError(f"Missing metadata key: {key}")

        # Validate scenes
        if not isinstance(scenes, list):
            raise ValueError("'scenes' must be a list")

        if len(scenes) < 6 or len(scenes) > 10:
            raise ValueError(f"Scene count must be 6-10, got {len(scenes)}")

        if metadata["scene_count"] != len(scenes):
            raise ValueError(
                f"scene_count mismatch: metadata says {metadata['scene_count']}, "
                f"but found {len(scenes)} scenes"
            )

        # Validate each scene
        total_duration = 0
        for scene in scenes:
            required_scene_keys = [
                "scene_number", "duration", "visual_description",
                "narration_text", "camera_movement"
            ]

            for key in required_scene_keys:
                if key not in scene:
                    raise ValueError(f"Scene missing key: {key}")

            # Validate duration
            duration = scene["duration"]
            if not isinstance(duration, (int, float)) or duration < 2 or duration > 5:
                raise ValueError(
                    f"Scene duration must be 2-5 seconds, got {duration}"
                )

            total_duration += duration

            # Validate camera movement
            camera = scene["camera_movement"]
            if camera not in VALID_CAMERA_MOVEMENTS:
                raise ValueError(
                    f"Invalid camera movement: {camera}. "
                    f"Must be one of {VALID_CAMERA_MOVEMENTS}"
                )

        # Check total duration is close to target
        duration_diff = abs(total_duration - target_duration)
        if duration_diff > 3:
            logger.warning(
                f"Duration mismatch: target={target_duration}s, "
                f"actual={total_duration}s (diff={duration_diff}s)"
            )

        logger.info(f"Validation passed: {len(scenes)} scenes, {total_duration}s total")

    def save_script(self, scene_data: dict, output_path: str) -> None:
        """
        Save scene script to JSON file.

        Args:
            scene_data: Scene data from generate_scenes()
            output_path: Path to save the file
        """
        output_path = str(output_path)

        # Create directory if needed
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(scene_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Scene script saved to: {output_path}")
