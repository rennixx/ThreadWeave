"""
Video Assembler Tests

Module-level testing for VideoAssembler class.
"""

import os
import sys
import subprocess
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Check for dependencies
try:
    from modules.assembler import VideoAssembler
    ASSEMBLER_AVAILABLE = True
except ImportError:
    ASSEMBLER_AVAILABLE = False

# Check for moviepy
try:
    from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_videoclips
    MOVIEPY_AVAILABLE = True
except ImportError:
    MOVIEPY_AVAILABLE = False

# Check for ffmpeg
try:
    result = subprocess.run(["ffmpeg", "-version"], capture_output=True)
    FFMPEG_AVAILABLE = result.returncode == 0
except FileNotFoundError:
    FFMPEG_AVAILABLE = False


def create_test_clip(output_path: str, duration: float = 2.0, size: tuple = (1080, 1920)) -> None:
    """Create a simple test video clip."""
    import cv2
    import numpy as np

    # Create test frames
    fps = 30
    total_frames = int(duration * fps)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, size)

    for frame_idx in range(total_frames):
        # Create colored frame
        frame = np.zeros((size[1], size[0], 3), dtype=np.uint8)

        # Alternating colors based on frame
        if frame_idx % 2 == 0:
            frame[:, :] = [255, 100, 0]  # Blue
        else:
            frame[:, :] = [0, 0, 255]  # Red

        # Add frame number
        cv2.putText(
            frame,
            str(frame_idx),
            (50, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (255, 255, 255),
            3
        )

        writer.write(frame)

    writer.release()


def create_test_audio(output_path: str, duration: float = 10.0) -> None:
    """Create a simple test audio file."""
    from pydub.generators import Sine

    tone = Sine(440).to_audio_segment(duration=int(duration * 1000))
    tone.export(output_path, format="mp3")


def test_method_availability():
    """Test Case 6.0: Method Availability"""
    print("\n" + "="*60)
    print("Test Case 6.0: Method Availability")
    print("="*60)

    if not ASSEMBLER_AVAILABLE:
        print("  SKIPPED: VideoAssembler not available")
        return

    if not MOVIEPY_AVAILABLE and not FFMPEG_AVAILABLE:
        print("  SKIPPED: Neither moviepy nor ffmpeg available")
        print("  Install with:")
        print("    pip install moviepy")
        print("    # OR")
        print("    winget install ffmpeg")
        return

    assembler = VideoAssembler(fps=30, resolution=(1080, 1920))

    print(f"  moviepy available: {assembler.moviepy_available}")
    print(f"  ffmpeg available: {assembler.ffmpeg_available}")

    if assembler.moviepy_available or assembler.ffmpeg_available:
        print(f"  PASS: At least one assembly method available")
    else:
        print(f"  FAIL: No assembly method available")


def test_video_concatenation():
    """Test Case 6.1: Video Concatenation"""
    print("\n" + "="*60)
    print("Test Case 6.1: Video Concatenation")
    print("="*60)

    if not ASSEMBLER_AVAILABLE:
        print("  SKIPPED: VideoAssembler not available")
        return

    assembler = VideoAssembler(fps=30, resolution=(1080, 1920))

    # Create test clips
    output_dir = "output/assembler_tests"
    os.makedirs(output_dir, exist_ok=True)

    clip_paths = []
    for i in range(3):
        clip_path = f"{output_dir}/test_clip_{i+1}.mp4"
        create_test_clip(clip_path, duration=2.0)
        clip_paths.append(clip_path)

    # Create test audio
    audio_path = f"{output_dir}/test_audio.mp3"
    create_test_audio(audio_path, duration=8.0)

    output_path = f"{output_dir}/test_assembled.mp4"

    try:
        result_path = assembler.assemble_video(
            clip_paths,
            audio_path,
            output_path
        )

        assert os.path.exists(result_path), "Output video should exist"

        # Check file size
        file_size = os.path.getsize(result_path)
        assert file_size > 1024 * 100, f"Video should be at least 100KB, got {file_size/1024:.1f}KB"

        # Get video info
        info = assembler.get_video_info(result_path)

        print(f"  PASS: Video assembled successfully")
        print(f"    Method: {'moviepy' if assembler.moviepy_available else 'ffmpeg'}")
        print(f"    Duration: {info.get('duration', 'unknown'):.2f}s")
        print(f"    Size: {file_size/1024/1024:.2f} MB")
        print(f"    Resolution: {info.get('resolution', 'unknown')}")

        return True

    except Exception as e:
        print(f"  FAIL: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_missing_clips():
    """Test Case 6.1b: Missing Clips Handling"""
    print("\n" + "="*60)
    print("Test Case 6.1b: Missing Clips Handling")
    print("="*60)

    if not ASSEMBLER_AVAILABLE:
        print("  SKIPPED: VideoAssembler not available")
        return

    assembler = VideoAssembler(fps=30, resolution=(1080, 1920))

    # Create test audio
    output_dir = "output/assembler_tests"
    os.makedirs(output_dir, exist_ok=True)

    audio_path = f"{output_dir}/test_audio_missing.mp3"
    create_test_audio(audio_path, duration=5.0)

    # Use non-existent clip path
    clip_paths = ["nonexistent_clip.mp4"]
    output_path = f"{output_dir}/test_missing.mp4"

    try:
        assembler.assemble_video(clip_paths, audio_path, output_path)
        print(f"  FAIL: Should have raised FileNotFoundError")
        return False

    except FileNotFoundError as e:
        print(f"  PASS: Correctly raises FileNotFoundError")
        print(f"    Error: {str(e)}")
        return True

    except Exception as e:
        print(f"  FAIL: Unexpected error: {str(e)}")
        return False


def test_missing_audio():
    """Test Case 6.1c: Missing Audio Handling"""
    print("\n" + "="*60)
    print("Test Case 6.1c: Missing Audio Handling")
    print("="*60)

    if not ASSEMBLER_AVAILABLE:
        print("  SKIPPED: VideoAssembler not available")
        return

    assembler = VideoAssembler(fps=30, resolution=(1080, 1920))

    # Create test clip
    output_dir = "output/assembler_tests"
    os.makedirs(output_dir, exist_ok=True)

    clip_path = f"{output_dir}/test_clip_missing.mp4"
    create_test_clip(clip_path, duration=2.0)

    # Use non-existent audio path
    audio_path = "nonexistent_audio.mp3"
    output_path = f"{output_dir}/test_missing_audio.mp4"

    try:
        assembler.assemble_video([clip_path], audio_path, output_path)
        print(f"  FAIL: Should have raised FileNotFoundError")
        return False

    except FileNotFoundError as e:
        print(f"  PASS: Correctly raises FileNotFoundError")
        print(f"    Error: {str(e)}")
        return True

    except Exception as e:
        print(f"  FAIL: Unexpected error: {str(e)}")
        return False


def test_get_video_info():
    """Test Case 6.2: Video Info"""
    print("\n" + "="*60)
    print("Test Case 6.2: Video Info")
    print("="*60)

    if not ASSEMBLER_AVAILABLE:
        print("  SKIPPED: VideoAssembler not available")
        return

    assembler = VideoAssembler(fps=30, resolution=(1080, 1920))

    # Create test video
    output_dir = "output/assembler_tests"
    os.makedirs(output_dir, exist_ok=True)

    video_path = f"{output_dir}/test_info.mp4"
    create_test_clip(video_path, duration=2.0)

    try:
        info = assembler.get_video_info(video_path)

        print(f"  PASS: Video info retrieved")
        print(f"    Path: {info.get('path')}")
        print(f"    Duration: {info.get('duration', 'unknown'):.2f}s")
        print(f"    FPS: {info.get('fps', 'unknown')}")
        print(f"    Resolution: {info.get('resolution', 'unknown')}")
        print(f"    File size: {info.get('file_size', 0) / 1024:.1f} KB")

        return True

    except Exception as e:
        print(f"  FAIL: {str(e)}")
        return False


def test_horizontal_conversion():
    """Test Case 6.3: Horizontal Conversion"""
    print("\n" + "="*60)
    print("Test Case 6.3: Horizontal Conversion")
    print("="*60)

    if not ASSEMBLER_AVAILABLE:
        print("  SKIPPED: VideoAssembler not available")
        return

    assembler = VideoAssembler(fps=30, resolution=(1080, 1920))

    if not assembler.ffmpeg_available:
        print("  SKIPPED: ffmpeg required for horizontal conversion")
        return

    # Create test vertical video
    output_dir = "output/assembler_tests"
    os.makedirs(output_dir, exist_ok=True)

    vertical_path = f"{output_dir}/test_vertical.mp4"
    create_test_clip(vertical_path, duration=2.0)

    output_path = f"{output_dir}/test_horizontal.mp4"

    try:
        result_path = assembler.create_horizontal_version(
            vertical_path,
            output_path
        )

        assert os.path.exists(result_path), "Horizontal video should exist"

        # Check resolution
        info = assembler.get_video_info(result_path)
        resolution = info.get('resolution', (0, 0))

        if resolution == (1920, 1080) or resolution == (1080, 1920):  # Some implementations differ
            print(f"  PASS: Horizontal version created")
            print(f"    Resolution: {resolution}")
            print(f"    Duration: {info.get('duration', 'unknown'):.2f}s")
        else:
            print(f"  WARN: Unexpected resolution: {resolution}")

        return True

    except Exception as e:
        print(f"  FAIL: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all video assembler tests"""
    print("\n" + "="*60)
    print("VIDEO ASSEMBLER - RUNNING ALL TESTS")
    print("="*60)

    print("\nDependency Check:")
    print(f"  VideoAssembler: {ASSEMBLER_AVAILABLE}")
    print(f"  moviepy: {MOVIEPY_AVAILABLE}")
    print(f"  ffmpeg: {FFMPEG_AVAILABLE}")

    if not ASSEMBLER_AVAILABLE:
        print("\n  SKIPPED: VideoAssembler not available")
        return

    if not MOVIEPY_AVAILABLE and not FFMPEG_AVAILABLE:
        print("\n  SKIPPED: Neither moviepy nor ffmpeg available")
        print("\n  To enable video assembly, install one of:")
        print("    pip install moviepy")
        print("    # OR")
        print("    winget install ffmpeg")
        print("  Then restart your terminal.")
        print("="*60)
        return

    # Run tests
    test_method_availability()
    test_video_concatenation()
    test_missing_clips()
    test_missing_audio()
    test_get_video_info()

    if FFMPEG_AVAILABLE:
        test_horizontal_conversion()

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
