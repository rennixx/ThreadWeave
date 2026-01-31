#!/usr/bin/env python3
"""
ThreadWeave - Thread-to-Animated-Video Generator MVP

Main entry point for the video generation pipeline.
"""

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# Import modules
from modules.scraper import ThreadScraper
from modules.scene_generator import SceneGenerator
from modules.image_gen import ImageGenerator
from modules.animator import Animator
from modules.audio_gen import AudioGenerator
from modules.assembler import VideoAssembler


def setup_logging(verbose: bool = False) -> None:
    """Set up logging configuration."""
    log_level = logging.DEBUG if verbose else logging.INFO

    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('pipeline.log', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )


def load_config():
    """Load configuration from environment variables and settings file."""
    from dotenv import load_dotenv
    import yaml

    # Load environment variables
    load_dotenv()

    config = {}

    # API Keys
    config['TWITTER_TOKEN'] = os.getenv("TWITTER_BEARER_TOKEN")
    config['OPENAI_KEY'] = os.getenv("OPENAI_API_KEY")
    config['ANTHROPIC_KEY'] = os.getenv("ANTHROPIC_API_KEY")
    config['ELEVENLABS_KEY'] = os.getenv("ELEVENLABS_API_KEY")

    # Load settings.yaml
    settings_path = "config/settings.yaml"
    if os.path.exists(settings_path):
        with open(settings_path, 'r') as f:
            settings = yaml.safe_load(f)
            config['ART_STYLE'] = settings.get("art_style", "minimalist geometric")
            config['DEFAULT_DURATION'] = settings.get("default_duration", 30)
            config['OUTPUT_RESOLUTION'] = settings.get("output_resolution", "1080x1920")
            config['VIDEO_FPS'] = settings.get("video_fps", 30)
            config['TTS_PROVIDER'] = settings.get("tts_provider", "edge")
            config['TTS_VOICE'] = settings.get("tts_voice", "en-US-AriaNeural")
            config['BACKGROUND_MUSIC'] = settings.get("background_music_path", "assets/background_music/default.mp3")
            config['OLLAMA_BASE_URL'] = settings.get("ollama_base_url", "http://localhost:11434")
            config['OLLAMA_MODEL'] = settings.get("ollama_model", "kimi-k2.5")
    else:
        # Defaults (use local options)
        config['ART_STYLE'] = "minimalist geometric"
        config['DEFAULT_DURATION'] = 30
        config['OUTPUT_RESOLUTION'] = "1080x1920"
        config['VIDEO_FPS'] = 30
        config['TTS_PROVIDER'] = "edge"
        config['TTS_VOICE'] = "en-US-AriaNeural"
        config['BACKGROUND_MUSIC'] = "assets/background_music/default.mp3"
        config['OLLAMA_BASE_URL'] = "http://localhost:11434"
        config['OLLAMA_MODEL'] = "kimi-k2.5"

    # Parse resolution
    width, height = map(int, config['OUTPUT_RESOLUTION'].split('x'))
    config['RESOLUTION'] = (width, height)

    # Check for Ollama model from env
    config['OLLAMA_MODEL'] = os.getenv("OLLAMA_MODEL", config.get('OLLAMA_MODEL', 'kimi-k2.5'))
    config['OLLAMA_BASE_URL'] = os.getenv("OLLAMA_BASE_URL", config.get('OLLAMA_BASE_URL', 'http://localhost:11434'))

    return config


def validate_config(config: dict) -> None:
    """Validate required configuration values."""
    # Check if we have any LLM option available
    has_api_key = config['OPENAI_KEY'] or config['ANTHROPIC_KEY']
    has_ollama = True  # Assume Ollama is available if user sets it up

    if not has_api_key and not has_ollama:
        raise ValueError(
            "No LLM option available. Set one of:\n"
            "  - OPENAI_API_KEY (for Openai)\n"
            "  - ANTHROPIC_API_KEY (for Claude)\n"
            "  - Or install Ollama for local LLM (see https://ollama.ai)"
        )

    if not config['TWITTER_TOKEN']:
        logging.warning(
            "TWITTER_BEARER_TOKEN not set. "
            "Thread scraping will use fallback method (may not work)."
        )


def print_step(step: int, total: int, message: str) -> None:
    """Print a pipeline step indicator."""
    print(f"\n📍 Step {step}/{total}: {message}")
    print("=" * 60)


def main(
    thread_url: str,
    style: str = None,
    duration: int = None,
    output_dir: str = "output/final",
    verbose: bool = False
) -> str:
    """
    Main pipeline orchestrator.

    Args:
        thread_url: URL of the Twitter thread to convert
        style: Optional art style override
        duration: Target video duration in seconds
        output_dir: Directory to save final video
        verbose: Enable verbose logging

    Returns:
        Path to the generated video file
    """
    # Setup
    setup_logging(verbose)
    logger = logging.getLogger(__name__)

    print("🎬 ThreadWeave - Thread-to-Video Generator")
    print("=" * 60)

    # Load and validate config
    config = load_config()
    validate_config(config)

    # Override config with command line args
    if style:
        config['ART_STYLE'] = style
    if duration:
        config['DURATION'] = duration
    else:
        config['DURATION'] = config.get('DEFAULT_DURATION', 30)

    print(f"Thread URL: {thread_url}")
    print(f"Target Duration: {config['DURATION']}s")
    print(f"Art Style: {config['ART_STYLE']}")
    print(f"Output Resolution: {config['OUTPUT_RESOLUTION']}")
    print(f"TTS Provider: {config['TTS_PROVIDER']}")

    # Create output directories
    for directory in [
        "output/threads",
        "output/scripts",
        "output/images",
        "output/clips",
        "output/audio",
        "output/final"
    ]:
        os.makedirs(directory, exist_ok=True)

    total_steps = 6
    thread_id = None

    try:
        # Step 1: Scrape Thread
        print_step(1, total_steps, "Scraping thread...")

        scraper = ThreadScraper(bearer_token=config['TWITTER_TOKEN'])
        thread_data = scraper.extract_thread(thread_url)

        thread_id = thread_data.get('thread_id', 'unknown')
        logger.info(f"Thread ID: {thread_id}")
        logger.info(f"Author: {thread_data.get('author', 'unknown')}")
        logger.info(f"Tweet count: {thread_data.get('tweet_count', 0)}")

        # Save thread data
        thread_path = f"output/threads/{thread_id}.json"
        scraper.save_thread(thread_data, thread_path)

        print(f"  ✅ Extracted {thread_data.get('tweet_count', 0)} tweets from {thread_data.get('author', 'unknown')}")

        # Step 2: Generate Scene Script
        print_step(2, total_steps, "Generating scene script...")

        # Determine which LLM to use (prefer local Ollama, then API keys)
        if os.getenv("OLLAMA_MODEL") or os.path.exists(config.get('OLLAMA_BASE_URL', '')):
            # Prefer Ollama if available
            llm_provider = "ollama"
            llm_key = None
            llm_base_url = config['OLLAMA_BASE_URL']
            print(f"  Using Ollama ({config['OLLAMA_MODEL']}) at {llm_base_url}")
        elif config['OPENAI_KEY']:
            llm_provider = "openai"
            llm_key = config['OPENAI_KEY']
            llm_base_url = None
            print(f"  Using OpenAI GPT")
        elif config['ANTHROPIC_KEY']:
            llm_provider = "anthropic"
            llm_key = config['ANTHROPIC_KEY']
            llm_base_url = None
            print(f"  Using Anthropic Claude")
        else:
            # Default to Ollama
            llm_provider = "ollama"
            llm_key = None
            llm_base_url = config['OLLAMA_BASE_URL']
            print(f"  Using Ollama ({config['OLLAMA_MODEL']}) at {llm_base_url}")

        scene_gen = SceneGenerator(api_key=llm_key, provider=llm_provider, base_url=llm_base_url)
        scene_script = scene_gen.generate_scenes(
            thread_data,
            art_style=config['ART_STYLE'],
            target_duration=config['DURATION']
        )

        # Save scene script
        script_path = f"output/scripts/{thread_id}_script.json"
        scene_gen.save_script(scene_script, script_path)

        print(f"  ✅ Generated {scene_script['metadata']['scene_count']} scenes")

        # Step 3: Generate Images
        print_step(3, total_steps, "Generating scene images...")

        image_gen = ImageGenerator(
            model_name="stabilityai/sdxl-turbo",
            device="cuda"  # Will auto-detect if CUDA not available
        )

        images_dir = f"output/images/{thread_id}"
        image_paths = image_gen.generate_scene_images(
            scene_script,
            output_dir=images_dir
        )

        # Unload model to free memory
        image_gen.unload_model()

        print(f"  ✅ Generated {len(image_paths)} images")

        # Step 4: Create Animations
        print_step(4, total_steps, "Creating animations...")

        animator = Animator(
            fps=config['VIDEO_FPS'],
            resolution=config['RESOLUTION']
        )

        clips_dir = f"output/clips/{thread_id}"
        clip_paths = animator.animate_scenes(
            scene_script,
            image_paths,
            output_dir=clips_dir
        )

        print(f"  ✅ Created {len(clip_paths)} animated clips")

        # Step 5: Generate Audio
        print_step(5, total_steps, "Generating audio...")

        # Determine TTS API key (only needed for OpenAI, ElevenLabs)
        tts_key = None
        if config['TTS_PROVIDER'] in ['openai', 'elevenlabs']:
            tts_key = config.get('OPENAI_KEY') or config.get('ELEVENLABS_KEY')

        audio_gen = AudioGenerator(
            tts_api_key=tts_key,
            tts_provider=config['TTS_PROVIDER'],
            voice=config['TTS_VOICE']
        )

        # Generate narration
        narration_path = f"output/audio/{thread_id}_narration.mp3"
        audio_gen.generate_narration(
            scene_script,
            output_path=narration_path
        )

        # Add background music
        final_audio_path = f"output/audio/{thread_id}_final.mp3"

        # Check if background music exists
        if os.path.exists(config['BACKGROUND_MUSIC']):
            audio_gen.add_background_music(
                narration_path,
                config['BACKGROUND_MUSIC'],
                final_audio_path
            )
        else:
            logger.warning(f"Background music not found: {config['BACKGROUND_MUSIC']}")
            logger.info("Using narration only (no background music)")
            import shutil
            shutil.copy(narration_path, final_audio_path)

        print(f"  ✅ Generated audio track")

        # Step 6: Assemble Final Video
        print_step(6, total_steps, "Assembling final video...")

        assembler = VideoAssembler(
            fps=config['VIDEO_FPS'],
            resolution=config['RESOLUTION']
        )

        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_video_path = f"{output_dir}/{thread_id}_{timestamp}.mp4"

        assembler.assemble_video(
            clip_paths,
            final_audio_path,
            final_video_path
        )

        # Get video info
        video_info = assembler.get_video_info(final_video_path)
        file_size_mb = video_info.get('file_size', 0) / (1024 * 1024)

        print(f"\n{'=' * 60}")
        print(f"✅ VIDEO GENERATED SUCCESSFULLY!")
        print(f"{'=' * 60}")
        print(f"📁 Output: {final_video_path}")
        print(f"⏱️  Duration: {video_info.get('duration', 0):.2f} seconds")
        print(f"📦 Size: {file_size_mb:.2f} MB")
        print(f"🎨 Resolution: {video_info.get('resolution', config['RESOLUTION'])}")
        print(f"{'=' * 60}")

        return final_video_path

    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        print("\n⚠️  Pipeline interrupted by user")
        raise

    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        print(f"\n❌ Pipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate animated video from Twitter thread",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py "https://twitter.com/user/status/123456789"
  python main.py "https://twitter.com/user/status/123456789" --style "watercolor painting"
  python main.py "https://twitter.com/user/status/123456789" --duration 60 --verbose

Environment variables (.env file):
  OPENAI_API_KEY         Required for scene generation and TTS
  ANTHROPIC_API_KEY      Alternative to OPENAI_API_KEY
  TWITTER_BEARER_TOKEN   Required for thread scraping
  ELEVENLABS_API_KEY     Optional: alternative TTS provider
        """
    )

    parser.add_argument(
        "thread_url",
        type=str,
        help="URL of the Twitter thread"
    )
    parser.add_argument(
        "--style",
        type=str,
        default=None,
        help="Art style override (e.g., 'watercolor painting', 'pixel art')"
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=None,
        help="Target duration in seconds (default: from config or 30)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/final",
        help="Directory to save final video (default: output/final)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    try:
        output_path = main(
            args.thread_url,
            style=args.style,
            duration=args.duration,
            output_dir=args.output_dir,
            verbose=args.verbose
        )
        sys.exit(0)

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(1)

    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)
