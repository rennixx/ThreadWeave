"""
Animator Tests

Module-level testing for Animator class.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.animator import Animator
import cv2
import numpy as np


def create_test_image(output_path: str, size: tuple = (1080, 1920)) -> None:
    """Create a simple test image."""
    # Create a colored test image
    image = np.zeros((size[1], size[0], 3), dtype=np.uint8)

    # Add some color patterns
    height, width = size[1], size[0]

    # Top half blue, bottom half red
    image[:height//2, :] = [255, 100, 0]  # Blue in BGR
    image[height//2:, :] = [0, 0, 255]  # Red in BGR

    # Add a green circle in the middle
    center = (width // 2, height // 2)
    radius = min(width, height) // 8
    cv2.circle(image, center, radius, (0, 255, 0), -1)

    # Add text
    text = "TEST"
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, text, (width//2 - 100, height//2 - 200), font, 3, (255, 255, 255), 5)

    # Save
    cv2.imwrite(output_path, image)


def test_all_camera_movements():
    """Test Case 4.1: All Camera Movements"""
    print("\n" + "="*60)
    print("Test Case 4.1: All Camera Movements")
    print("="*60)

    animator = Animator(fps=30, resolution=(1080, 1920))

    # Create test image
    test_image_path = "output/test_images/test_animator_source.png"
    os.makedirs("output/test_images", exist_ok=True)
    create_test_image(test_image_path)

    output_dir = "output/animation_tests"
    os.makedirs(output_dir, exist_ok=True)

    movements = ["static", "zoom_in", "zoom_out", "pan_left", "pan_right"]
    results = []

    for movement in movements:
        scene = {
            "duration": 3.0,
            "camera_movement": movement
        }

        output_path = f"{output_dir}/test_{movement}.mp4"

        try:
            animator._create_clip(test_image_path, scene, output_path)

            # Verify clip exists
            assert os.path.exists(output_path), f"Clip should exist: {output_path}"

            # Check file size
            file_size = os.path.getsize(output_path)
            assert file_size > 0, f"Clip should not be empty: {output_path}"

            # Get clip info
            info = animator.get_clip_info(output_path)

            print(f"  PASS: {movement:10s} - {file_size/1024:6.1f} KB, "
                  f"{info['frame_count']} frames, {info['duration']:.2f}s")

            results.append(True)

        except Exception as e:
            print(f"  FAIL: {movement} - {str(e)}")
            results.append(False)

    # Cleanup test image
    if os.path.exists(test_image_path):
        os.remove(test_image_path)

    if all(results):
        print(f"\n  PASS: All {len(movements)} camera movements work")
    else:
        print(f"\n  FAIL: {results.count(False)} movements failed")

    return all(results)


def test_duration_accuracy():
    """Test Case 4.2: Duration Accuracy"""
    print("\n" + "="*60)
    print("Test Case 4.2: Duration Accuracy")
    print("="*60)

    animator = Animator(fps=30, resolution=(1080, 1920))

    # Create test image
    test_image_path = "output/test_images/test_duration_source.png"
    os.makedirs("output/test_images", exist_ok=True)
    create_test_image(test_image_path)

    test_durations = [2.0, 3.5, 5.0]
    results = []

    for target_duration in test_durations:
        output_path = f"output/animation_tests/duration_{target_duration}.mp4"

        scene = {
            "duration": target_duration,
            "camera_movement": "static"
        }

        try:
            animator._create_clip(test_image_path, scene, output_path)

            # Check actual duration
            info = animator.get_clip_info(output_path)
            actual_duration = info["duration"]

            # Allow 0.1 second tolerance
            diff = abs(actual_duration - target_duration)

            if diff < 0.1:
                print(f"  PASS: {target_duration}s -> {actual_duration:.2f}s (diff: {diff:.3f}s)")
                results.append(True)
            else:
                print(f"  FAIL: {target_duration}s -> {actual_duration:.2f}s (diff: {diff:.3f}s)")
                results.append(False)

        except Exception as e:
            print(f"  FAIL: {target_duration}s - {str(e)}")
            results.append(False)

    # Cleanup
    if os.path.exists(test_image_path):
        os.remove(test_image_path)

    if all(results):
        print(f"\n  PASS: All duration tests passed")
    else:
        print(f"\n  FAIL: {results.count(False)} duration tests failed")

    return all(results)


def test_invalid_movement():
    """Test Case 4.3a: Invalid Movement Handling"""
    print("\n" + "="*60)
    print("Test Case 4.3a: Invalid Movement Handling")
    print("="*60)

    animator = Animator(fps=30, resolution=(1080, 1920))

    # Create test image
    test_image_path = "output/test_images/test_invalid_source.png"
    os.makedirs("output/test_images", exist_ok=True)
    create_test_image(test_image_path)

    # Test invalid movement (should default to static)
    scene = {
        "duration": 2.0,
        "camera_movement": "invalid_movement_xyz"
    }

    output_path = "output/animation_tests/test_invalid.mp4"

    try:
        animator._create_clip(test_image_path, scene, output_path)

        # Should still create a video (defaults to static)
        assert os.path.exists(output_path), "Clip should be created even with invalid movement"

        info = animator.get_clip_info(output_path)

        print(f"  PASS: Invalid movement handled correctly")
        print(f"    Created clip: {info['duration']}s, {info['frame_count']} frames")
        print(f"    (Should default to 'static' movement)")

        # Cleanup
        os.remove(output_path)

        return True

    except Exception as e:
        print(f"  FAIL: {str(e)}")
        return False

    finally:
        if os.path.exists(test_image_path):
            os.remove(test_image_path)


def test_scene_validation():
    """Test Case 4.3b: Scene Validation"""
    print("\n" + "="*60)
    print("Test Case 4.3b: Scene Validation")
    print("="*60)

    animator = Animator(fps=30, resolution=(1080, 1920))

    # Create test image
    test_image_path = "output/test_images/test_validation_source.png"
    os.makedirs("output/test_images", exist_ok=True)
    create_test_image(test_image_path)

    test_cases = [
        {
            "name": "Valid scene",
            "scene": {"duration": 3.0, "camera_movement": "zoom_in"},
            "should_pass": True
        },
        {
            "name": "Negative duration",
            "scene": {"duration": -1.0, "camera_movement": "static"},
            "should_pass": False
        },
        {
            "name": "Zero duration",
            "scene": {"duration": 0, "camera_movement": "static"},
            "should_pass": False
        }
    ]

    results = []

    for case in test_cases:
        output_path = f"output/animation_tests/test_validation_{case['name'].replace(' ', '_')}.mp4"

        try:
            animator._create_clip(test_image_path, case["scene"], output_path)

            if case["should_pass"]:
                print(f"  PASS: {case['name']}")
                results.append(True)
            else:
                print(f"  FAIL: {case['name']} - should have raised error")
                results.append(False)

            # Cleanup
            if os.path.exists(output_path):
                os.remove(output_path)

        except (ValueError, RuntimeError) as e:
            if not case["should_pass"]:
                print(f"  PASS: {case['name']} - correctly raised error")
                results.append(True)
            else:
                print(f"  FAIL: {case['name']} - unexpected error: {e}")
                results.append(False)

    # Cleanup
    if os.path.exists(test_image_path):
        os.remove(test_image_path)

    if all(results):
        print(f"\n  PASS: All validation tests passed")
    else:
        print(f"\n  FAIL: {results.count(False)} validation tests failed")

    return all(results)


def test_animate_scenes():
    """Test Case 4.4: Animate Scenes (Batch)"""
    print("\n" + "="*60)
    print("Test Case 4.4: Animate Scenes (Batch)")
    print("="*60)

    animator = Animator(fps=30, resolution=(1080, 1920))

    # Create test images
    output_dir = "output/animation_tests/batch"
    os.makedirs(output_dir, exist_ok=True)

    image_paths = []
    for i in range(3):
        img_path = f"output/test_images/batch_scene_{i+1}.png"
        os.makedirs("output/test_images", exist_ok=True)
        create_test_image(img_path)
        image_paths.append(img_path)

    scene_script = {
        "metadata": {"art_style": "test"},
        "scenes": [
            {"scene_number": 1, "duration": 2.5, "camera_movement": "static"},
            {"scene_number": 2, "duration": 3.0, "camera_movement": "zoom_in"},
            {"scene_number": 3, "duration": 2.0, "camera_movement": "pan_right"}
        ]
    }

    try:
        clip_paths = animator.animate_scenes(scene_script, image_paths, output_dir)

        assert len(clip_paths) == 3, f"Should create 3 clips, got {len(clip_paths)}"

        # Verify all clips exist
        for path in clip_paths:
            assert os.path.exists(path), f"Clip should exist: {path}"

        print(f"  PASS: Created {len(clip_paths)} clips")
        for path in clip_paths:
            info = animator.get_clip_info(path)
            print(f"    {os.path.basename(path)}: {info['duration']:.2f}s, {info['frame_count']} frames")

        # Cleanup
        for img in image_paths:
            if os.path.exists(img):
                os.remove(img)

        return True

    except Exception as e:
        print(f"  FAIL: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all animator tests"""
    print("\n" + "="*60)
    print("ANIMATOR - RUNNING ALL TESTS")
    print("="*60)

    # Run all tests
    results = []
    results.append(("All Camera Movements", test_all_camera_movements()))
    results.append(("Duration Accuracy", test_duration_accuracy()))
    results.append(("Invalid Movement", test_invalid_movement()))
    results.append(("Scene Validation", test_scene_validation()))
    results.append(("Animate Scenes Batch", test_animate_scenes()))

    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {status}: {name}")

    all_passed = all(r[1] for r in results)

    if all_passed:
        print(f"\n  All tests passed!")
    else:
        print(f"\n  {sum(not r[1] for r in results)} test(s) failed")

    print("="*60)


if __name__ == "__main__":
    # Set UTF-8 encoding for Windows console
    if sys.platform == "win32":
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

    run_all_tests()
