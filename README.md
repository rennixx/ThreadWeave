# ThreadWeave

**Thread-to-Animated-Video Generator MVP**

Converts Twitter/X threads into engaging animated videos for TikTok, Reels, and Shorts.

## Features

- Thread scraping from Twitter/X URLs
- LLM-powered scene generation
- AI-generated consistent visuals (Stable Diffusion XL)
- Smooth camera animations
- Text-to-speech narration
- Background music mixing
- Vertical video output (1080x1920)

## Quick Start

### 1. Install Dependencies

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### 2. Configure API Keys

```bash
# Copy example env file
cp .env.example .env

# Edit .env and add your API keys:
# - OPENAI_API_KEY (required)
# - TWITTER_BEARER_TOKEN (required)
# - ANTHROPIC_API_KEY (optional)
# - ELEVENLABS_API_KEY (optional)
```

### 3. Run

```bash
python main.py "https://twitter.com/user/status/123456789"
```

## Requirements

- Python 3.10+
- CUDA-capable GPU (NVIDIA RTX 4070 or similar recommended)
- 12GB+ VRAM

## Project Structure

```
ThreadWeave/
├── config/                  # Configuration files
├── modules/                 # Core modules
│   ├── scraper.py          # Thread scraper
│   ├── scene_generator.py  # LLM scene generation
│   ├── image_gen.py        # Image generation
│   ├── animator.py         # Video animation
│   ├── audio_gen.py        # Audio generation
│   └── assembler.py        # Video assembly
├── output/                  # Generated content
├── assets/                  # Background music
└── main.py                 # Main pipeline
```

## Configuration

Edit `config/settings.yaml` to customize:
- Video resolution and quality
- Art style
- Scene count and duration
- Audio settings

## License

MIT License
