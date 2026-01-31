"""
Audio Generator Module

Generates TTS narration and mixes with background music.
"""

import logging
import os
from typing import Optional

from openai import OpenAI
import edge_tts
from pydub import AudioSegment
from pydub.generators import Sine
from tqdm import tqdm

logger = logging.getLogger(__name__)


class AudioGenerator:
    """
    Generates audio narration using TTS and mixes with background music.

    Supports OpenAI TTS, Edge TTS (free), ElevenLabs, and local TTS.
    """

    def __init__(
        self,
        tts_api_key: Optional[str] = None,
        tts_provider: str = "edge",
        voice: str = "en-US-AriaNeural"
    ):
        """
        Initialize the audio generator.

        Args:
            tts_api_key: API key for TTS service (not needed for Edge TTS)
            tts_provider: "openai", "edge", "elevenlabs", or "local"
            voice: Voice name
                - OpenAI: alloy, echo, fable, onyx, nova, shimmer
                - Edge TTS: en-US-AriaNeural, en-US-GuyNeural, en-GB-SoniaNeural, etc.
        """
        self.tts_provider = tts_provider.lower()
        self.voice = voice
        self.client = None

        # Audio settings
        self.sample_rate = 44100
        self.channels = 2
        self.bitrate = "192k"

        # Volume settings (from config)
        self._load_volume_settings()

        # Initialize TTS client
        if self.tts_provider == "openai":
            if not tts_api_key:
                logger.warning("OpenAI API key not provided")
            else:
                self.client = OpenAI(api_key=tts_api_key)
                logger.info(f"AudioGenerator initialized with OpenAI TTS (voice: {voice})")
        elif self.tts_provider == "edge":
            # Edge TTS doesn't need initialization
            logger.info(f"AudioGenerator initialized with Edge TTS (voice: {voice})")
        elif self.tts_provider == "elevenlabs":
            if not tts_api_key:
                logger.warning("ElevenLabs API key not provided")
            else:
                # ElevenLabs initialization
                try:
                    import elevenlabs
                    self.client = elevenlabs.ElevenLabs(api_key=tts_api_key)
                    logger.info(f"AudioGenerator initialized with ElevenLabs")
                except ImportError:
                    logger.warning("ElevenLabs not installed. Run: pip install elevenlabs")
        elif self.tts_provider == "local":
            logger.info("AudioGenerator initialized with local TTS")
        else:
            logger.warning(f"Unknown TTS provider: {tts_provider}. Using fallback.")

    def _load_volume_settings(self) -> None:
        """Load volume settings from config file."""
        config_path = "config/settings.yaml"

        default_narration_volume = 0  # dB
        default_music_volume = -22  # dB

        if os.path.exists(config_path):
            import yaml
            try:
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                    self.narration_volume = config.get("audio_volume_narration", default_narration_volume)
                    self.music_volume = config.get("audio_volume_music", default_music_volume)
            except Exception as e:
                logger.warning(f"Failed to load config: {e}. Using defaults.")
                self.narration_volume = default_narration_volume
                self.music_volume = default_music_volume
        else:
            self.narration_volume = default_narration_volume
            self.music_volume = default_music_volume

        logger.debug(f"Volume settings: narration={self.narration_volume}dB, music={self.music_volume}dB")

    def generate_narration(
        self,
        scene_script: dict,
        output_path: str
    ) -> str:
        """
        Generate voiceover narration from scene script.

        Args:
            scene_script: Scene script JSON with narration_text for each scene
            output_path: Path to save the narration audio file

        Returns:
            str: Path to the generated narration file

        Raises:
            ValueError: If scene_script is invalid
            RuntimeError: If TTS generation fails
        """
        if not scene_script or "scenes" not in scene_script:
            raise ValueError("Invalid scene_script: missing 'scenes'")

        scenes = scene_script["scenes"]

        if not scenes:
            raise ValueError("No scenes to generate narration for")

        # Create output directory
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        logger.info(f"Generating narration for {len(scenes)} scenes...")

        # Generate audio for each scene
        segments = []

        for scene in tqdm(scenes, desc="Generating TTS"):
            narration_text = scene.get("narration_text", "")
            duration = scene.get("duration", 3.0)

            if not narration_text:
                logger.warning(f"Scene {scene.get('scene_number', '?')} has no narration text")
                # Add silence for scene duration
                silence = AudioSegment.silent(duration=int(duration * 1000))
                segments.append(silence)
                continue

            try:
                segment = self._generate_speech_segment(narration_text, duration)
                segments.append(segment)

            except Exception as e:
                logger.error(f"Failed to generate TTS for scene {scene.get('scene_number', '?')}: {e}")
                # Add silence as fallback
                silence = AudioSegment.silent(duration=int(duration * 1000))
                segments.append(silence)

        # Concatenate all segments
        if not segments:
            raise RuntimeError("No audio segments generated")

        full_narration = segments[0]
        for segment in segments[1:]:
            # Add a small crossfade to prevent clicks
            full_narration = full_narration.append(segment, crossfade=50)

        # Adjust volume
        full_narration = full_narration + self.narration_volume

        # Export
        full_narration.export(
            output_path,
            format="mp3",
            bitrate=self.bitrate
        )

        logger.info(f"Narration saved to: {output_path}")

        return output_path

    def _generate_speech_segment(self, text: str, duration: float) -> AudioSegment:
        """
        Generate TTS audio for a single scene's narration.

        Args:
            text: Narration text
            duration: Target duration in seconds

        Returns:
            AudioSegment: Generated audio segment
        """
        if not text.strip():
            return AudioSegment.silent(duration=int(duration * 1000))

        if self.tts_provider == "openai" and self.client:
            return self._generate_openai_tts(text, duration)

        elif self.tts_provider == "elevenlabs" and self.client:
            return self._generate_elevenlabs_tts(text, duration)

        else:
            # Fallback to silence with warning
            logger.warning("TTS not available, using silence")
            return AudioSegment.silent(duration=int(duration * 1000))

    def _generate_openai_tts(self, text: str, target_duration: float) -> AudioSegment:
        """Generate speech using OpenAI TTS API."""
        try:
            response = self.client.audio.speech.create(
                model="tts-1",
                voice=self.voice,
                input=text,
                speed=1.0
            )

            # Save to temp file and load
            temp_path = "temp_tts.mp3"
            with open(temp_path, 'wb') as f:
                f.write(response.content)

            audio = AudioSegment.from_mp3(temp_path)
            os.remove(temp_path)

            # Adjust duration if needed
            actual_duration = len(audio) / 1000  # Convert to seconds

            if abs(actual_duration - target_duration) > 0.5:
                # Adjust speed to match target duration
                if actual_duration < target_duration:
                    # Audio is too short, slow it down
                    ratio = actual_duration / target_duration
                    audio = audio._spawn(audio.raw_data, overrides={
                        "frame_rate": int(audio.frame_rate * ratio)
                    }).set_frame_rate(audio.frame_rate)
                else:
                    # Audio is too long, speed it up (max 1.2x)
                    ratio = max(actual_duration / target_duration, 1.0)
                    if ratio > 1.2:
                        ratio = 1.2
                    audio = audio._spawn(audio.raw_data, overrides={
                        "frame_rate": int(audio.frame_rate / ratio)
                    }).set_frame_rate(audio.frame_rate)

            return audio

        except Exception as e:
            logger.error(f"OpenAI TTS failed: {e}")
            raise

    def _generate_elevenlabs_tts(self, text: str, target_duration: float) -> AudioSegment:
        """Generate speech using ElevenLabs API."""
        try:
            import elevenlabs

            audio = elevenlabs.generate(
                text=text,
                voice=self.voice,
                model="eleven_monolingual_v1"
            )

            # Save to temp file and load
            temp_path = "temp_tts.mp3"
            elevenlabs.save(audio, temp_path)

            segment = AudioSegment.from_mp3(temp_path)
            os.remove(temp_path)

            return segment

        except Exception as e:
            logger.error(f"ElevenLabs TTS failed: {e}")
            raise

    def add_background_music(
        self,
        narration_path: str,
        music_path: str,
        output_path: str,
        fade_duration: float = 2.0
    ) -> str:
        """
        Mix narration with background music.

        Args:
            narration_path: Path to narration audio file
            music_path: Path to background music file
            output_path: Path to save mixed audio
            fade_duration: Fade in/out duration for music (seconds)

        Returns:
            str: Path to the mixed audio file

        Raises:
            FileNotFoundError: If input files don't exist
        """
        # Validate inputs
        if not os.path.exists(narration_path):
            raise FileNotFoundError(f"Narration not found: {narration_path}")

        if not os.path.exists(music_path):
            logger.warning(f"Background music not found: {music_path}. Using narration only.")
            # Just copy narration to output
            import shutil
            shutil.copy(narration_path, output_path)
            return output_path

        # Load audio files
        logger.info(f"Mixing narration with background music...")

        try:
            narration = AudioSegment.from_file(narration_path)
            music = AudioSegment.from_file(music_path)
        except Exception as e:
            logger.error(f"Failed to load audio: {e}")
            raise

        # Adjust volumes
        narration = narration + self.narration_volume
        music = music + self.music_volume

        # Loop music to match or exceed narration length
        narration_length = len(narration)
        music_length = len(music)

        if music_length < narration_length:
            # Calculate how many loops needed
            loops_needed = int(narration_length / music_length) + 1
            music = music * loops_needed

        # Trim music to narration length
        music = music[:narration_length]

        # Apply fade in/out to music
        if fade_duration > 0:
            fade_ms = int(fade_duration * 1000)
            music = music.fade_in(fade_ms).fade_out(fade_ms)

        # Overlay music on narration
        final_audio = narration.overlay(music)

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        # Export
        final_audio.export(
            output_path,
            format="mp3",
            bitrate=self.bitrate
        )

        logger.info(f"Mixed audio saved to: {output_path}")

        return output_path

    def get_audio_info(self, audio_path: str) -> dict:
        """
        Get information about an audio file.

        Args:
            audio_path: Path to audio file

        Returns:
            dict: Audio information (duration, channels, sample_rate, frame_rate, file_size)
        """
        audio = AudioSegment.from_file(audio_path)

        return {
            "path": audio_path,
            "duration_seconds": len(audio) / 1000,
            "channels": audio.channels,
            "sample_rate": audio.frame_rate,
            "frame_rate": audio.frame_rate,
            "frame_width": audio.sample_width * 8,
            "file_size_bytes": os.path.getsize(audio_path)
        }

    def create_silence(self, duration_ms: int) -> AudioSegment:
        """
        Create a silence audio segment.

        Args:
            duration_ms: Duration in milliseconds

        Returns:
            AudioSegment: Silence segment
        """
        return AudioSegment.silent(duration=duration_ms)
