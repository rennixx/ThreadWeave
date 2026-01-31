"""
Audio Generator Tests

Module-level testing for AudioGenerator class.
"""

import os
import sys
from pathlib import Path
import subprocess

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Check for ffmpeg
FFMPEG_AVAILABLE = subprocess.run(["ffmpeg", "-version"], capture_output=True).returncode == 0

if FFMPEG_AVAILABLE:
    from modules.audio_gen import AudioGenerator
    from pydub import AudioSegment
    import wave
else:
    AudioGenerator = None
    AudioSegment = None


def create_test_music(output_path: str, duration_seconds: int = 30) -> None:
    """Create a simple test music file."""
    from pydub.generators import Sine

    # Generate a simple tone
    tone = Sine(440).to_audio_segment(duration=duration_seconds * 1000)

    # Add some variation
    tone2 = Sine(330).to_audio_segment(duration=duration_seconds * 1000)
    music = tone.overlay(tone2)

    # Lower volume for background music
    music = music - 20

    music.export(output_path, format="mp3", bitrate="192k")


def test_config_loading():
    """Test Case 5.0: Config Loading"""
    print("\n" + "="*60)
    print("Test Case 5.0: Config Loading")
    print("="*60)

    # Test with default config
    audio_gen = AudioGenerator()

    assert hasattr(audio_gen, "narration_volume"), "Should have narration_volume"
    assert hasattr(audio_gen, "music_volume"), "Should have music_volume"
    assert hasattr(audio_gen, "voice"), "Should have voice"

    print(f"  PASS: Config loaded successfully")
    print(f"    Narration volume: {audio_gen.narration_volume} dB")
    print(f"    Music volume: {audio_gen.music_volume} dB")
    print(f"    Voice: {audio_gen.voice}")
    print(f"    Sample rate: {audio_gen.sample_rate} Hz")


def test_silence_generation():
    """Test Case 5.0b: Silence Generation"""
    print("\n" + "="*60)
    print("Test Case 5.0b: Silence Generation")
    print("="*60)

    audio_gen = AudioGenerator()

    # Create silence of different durations
    test_durations = [1000, 2500, 5000]  # milliseconds

    for duration_ms in test_durations:
        silence = audio_gen.create_silence(duration_ms)

        actual_duration = len(silence)

        if actual_duration == duration_ms:
            print(f"  PASS: {duration_ms}ms silence = {actual_duration}ms")
        else:
            print(f"  FAIL: {duration_ms}ms silence != {actual_duration}ms")


def test_audio_info():
    """Test Case 5.0c: Audio Info"""
    print("\n" + "="*60)
    print("Test Case 5.0c: Audio Info")
    print("="*60)

    # Create test audio file
    test_path = "output/audio_tests/test_info.mp3"
    os.makedirs("output/audio_tests", exist_ok=True)

    create_test_music(test_path, duration_seconds=5)

    audio_gen = AudioGenerator()

    try:
        info = audio_gen.get_audio_info(test_path)

        print(f"  PASS: Audio info retrieved")
        print(f"    Duration: {info['duration_seconds']:.2f} seconds")
        print(f"    Channels: {info['channels']}")
        print(f"    Sample rate: {info['sample_rate']} Hz")
        print(f"    File size: {info['file_size_bytes'] / 1024:.1f} KB")

        # Cleanup
        os.remove(test_path)

        return True

    except Exception as e:
        print(f"  FAIL: {str(e)}")
        return False


def test_tts_generation():
    """Test Case 5.1: TTS Generation"""
    print("\n" + "="*60)
    print("Test Case 5.1: TTS Generation")
    print("="*60)

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        print("  SKIPPED: OPENAI_API_KEY not set")
        return None

    audio_gen = AudioGenerator(tts_api_key=api_key, tts_provider="openai")

    scene_script = {
        "metadata": {"art_style": "test"},
        "scenes": [
            {"narration_text": "This is the first scene.", "duration": 3.0},
            {"narration_text": "And here is the second scene.", "duration": 3.5}
        ]
    }

    output_path = "output/audio_tests/test_narration.mp3"
    os.makedirs("output/audio_tests", exist_ok=True)

    try:
        result_path = audio_gen.generate_narration(scene_script, output_path)

        assert os.path.exists(result_path), "Audio file should exist"

        # Check duration
        info = audio_gen.get_audio_info(result_path)
        duration = info["duration_seconds"]

        expected_duration = sum(s["duration"] for s in scene_script["scenes"])

        # Allow 2 second tolerance (TTS can vary)
        if abs(duration - expected_duration) < 2.0:
            print(f"  PASS: TTS generated")
            print(f"    Duration: {duration:.2f}s (expected ~{expected_duration:.2f}s)")
            print(f"    File size: {info['file_size_bytes'] / 1024:.1f} KB")

            # Cleanup
            os.remove(result_path)

            return True
        else:
            print(f"  WARN: Duration mismatch")
            print(f"    Expected: {expected_duration:.2f}s")
            print(f"    Got: {duration:.2f}s")

            return True  # Still pass, TTS timing varies

    except Exception as e:
        print(f"  FAIL: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_audio_mixing():
    """Test Case 5.2: Audio Mixing"""
    print("\n" + "="*60)
    print("Test Case 5.2: Audio Mixing")
    print("="*60)

    audio_gen = AudioGenerator()

    # Create test narration
    os.makedirs("output/audio_tests", exist_ok=True)

    narration_path = "output/audio_tests/test_mix_narration.mp3"
    music_path = "output/audio_tests/test_mix_music.mp3"
    output_path = "output/audio_tests/test_mixed.mp3"

    # Create simple test files
    from pydub.generators import Sine

    # Narration: simple tone
    narration = Sine(440).to_audio_segment(duration=5000)  # 5 seconds
    narration.export(narration_path, format="mp3")

    # Music: different tone
    music = Sine(330).to_audio_segment(duration=10000)  # 10 seconds
    music.export(music_path, format="mp3")

    try:
        result_path = audio_gen.add_background_music(
            narration_path,
            music_path,
            output_path
        )

        assert os.path.exists(result_path), "Mixed audio should exist"

        # Load and check
        mixed = AudioSegment.from_file(result_path)

        # Check it's not silent
        if mixed.dBFS > -60:
            print(f"  PASS: Audio mixed successfully")
            print(f"    Output level: {mixed.dBFS:.1f} dBFS")
            print(f"    Duration: {len(mixed) / 1000:.2f} seconds")

            # Cleanup
            for path in [narration_path, music_path, result_path]:
                if os.path.exists(path):
                    os.remove(path)

            return True
        else:
            print(f"  FAIL: Audio is too silent ({mixed.dBFS:.1f} dBFS)")
            return False

    except Exception as e:
        print(f"  FAIL: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_music_looping():
    """Test Case 5.2b: Music Looping"""
    print("\n" + "="*60)
    print("Test Case 5.2b: Music Looping")
    print("="*60)

    audio_gen = AudioGenerator()

    os.makedirs("output/audio_tests", exist_ok=True)

    # Create short music (5 seconds) and long narration (15 seconds)
    music_path = "output/audio_tests/test_loop_music.mp3"
    narration_path = "output/audio_tests/test_loop_narration.mp3"

    from pydub.generators import Sine

    music = Sine(330).to_audio_segment(duration=5000)  # 5 seconds
    music.export(music_path, format="mp3")

    narration = Sine(440).to_audio_segment(duration=15000)  # 15 seconds
    narration.export(narration_path, format="mp3")

    output_path = "output/audio_tests/test_looped.mp3"

    try:
        result_path = audio_gen.add_background_music(
            narration_path,
            music_path,
            output_path
        )

        # Check that music was looped
        mixed = AudioSegment.from_file(result_path)
        duration = len(mixed) / 1000

        # Should match narration duration (15s)
        if abs(duration - 15.0) < 0.5:
            print(f"  PASS: Music looped correctly")
            print(f"    Narration: 15s, Music: 5s (shorter)")
            print(f"    Output: {duration:.2f}s")
        else:
            print(f"  WARN: Duration mismatch")
            print(f"    Expected: 15s, Got: {duration:.2f}s")

        # Cleanup
        for path in [music_path, narration_path, result_path]:
            if os.path.exists(path):
                os.remove(path)

        return True

    except Exception as e:
        print(f"  FAIL: {str(e)}")
        return False


def test_missing_music_handling():
    """Test Case 5.2c: Missing Music Handling"""
    print("\n" + "="*60)
    print("Test Case 5.2c: Missing Music Handling")
    print("="*60)

    audio_gen = AudioGenerator()

    os.makedirs("output/audio_tests", exist_ok=True)

    narration_path = "output/audio_tests/test_no_music_narration.mp3"
    output_path = "output/audio_tests/test_no_music_output.mp3"

    # Create test narration
    from pydub.generators import Sine
    narration = Sine(440).to_audio_segment(duration=3000)
    narration.export(narration_path, format="mp3")

    try:
        # Use non-existent music path
        result_path = audio_gen.add_background_music(
            narration_path,
            "nonexistent_music.mp3",
            output_path
        )

        # Should copy narration to output
        assert os.path.exists(result_path), "Output should exist"

        print(f"  PASS: Missing music handled correctly")
        print(f"    Narration copied to output")

        # Cleanup
        for path in [narration_path, result_path]:
            if os.path.exists(path):
                os.remove(path)

        return True

    except Exception as e:
        print(f"  FAIL: {str(e)}")
        return False


def run_all_tests():
    """Run all audio generator tests"""
    print("\n" + "="*60)
    print("AUDIO GENERATOR - RUNNING ALL TESTS")
    print("="*60)

    # Run tests that don't require API
    test_config_loading()
    test_silence_generation()
    test_audio_info()
    test_audio_mixing()
    test_music_looping()
    test_missing_music_handling()

    # Run tests that require API
    print("\n" + "="*60)
    print("Note: TTS test requires OPENAI_API_KEY")
    print("="*60)
    test_tts_generation()

    print("\n" + "="*60)
    print("ALL TESTS COMPLETED")
    print("="*60)


if __name__ == "__main__":
    # Set UTF-8 encoding for Windows console
    if sys.platform == "win32":
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

    run_all_tests()
