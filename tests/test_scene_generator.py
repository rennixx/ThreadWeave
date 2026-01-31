"""
Scene Generator Tests

Module-level testing for SceneGenerator class.
"""

import json
import os
import sys
from pathlib import Path
from unittest.mock import Mock, patch

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.scene_generator import SceneGenerator, VALID_CAMERA_MOVEMENTS


def test_valid_json_output():
    """Test Case 2.1: Valid JSON Output"""
    print("\n" + "="*60)
    print("Test Case 2.1: Valid JSON Output")
    print("="*60)

    # Load API key from environment
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")

    if not api_key:
        print("  SKIPPED: OPENAI_API_KEY or ANTHROPIC_API_KEY not set")
        return None

    provider = "openai" if os.getenv("OPENAI_API_KEY") else "anthropic"

    try:
        scene_gen = SceneGenerator(api_key=api_key, provider=provider)

        # Sample thread data
        thread_data = {
            "thread_id": "test123",
            "combined_text": (
                "Artificial intelligence is transforming how we work and create. "
                "Machine learning models can now generate images, text, and even code. "
                "This democratizes creativity in unprecedented ways. "
                "However, we must also consider the ethical implications. "
                "The future of AI is both exciting and uncertain."
            )
        }

        result = scene_gen.generate_scenes(
            thread_data,
            art_style="minimalist geometric",
            target_duration=30
        )

        # Validate structure
        assert "metadata" in result, "Should have metadata"
        assert "scenes" in result, "Should have scenes array"
        assert isinstance(result["scenes"], list), "Scenes should be a list"

        # Validate metadata
        metadata = result["metadata"]
        assert "total_duration" in metadata, "Should have total_duration"
        assert "scene_count" in metadata, "Should have scene_count"

        # Validate scenes
        assert len(result["scenes"]) >= 6, "Should have at least 6 scenes"
        assert len(result["scenes"]) <= 10, "Should have at most 10 scenes"

        # Validate each scene
        for scene in result["scenes"]:
            assert "scene_number" in scene, "Scene should have number"
            assert "duration" in scene, "Scene should have duration"
            assert "visual_description" in scene, "Scene should have visual description"
            assert "narration_text" in scene, "Scene should have narration"
            assert "camera_movement" in scene, "Scene should have camera movement"

            # Validate duration
            assert 2 <= scene["duration"] <= 5, \
                f"Scene duration {scene['duration']} should be 2-5 seconds"

            # Validate camera movement
            assert scene["camera_movement"] in VALID_CAMERA_MOVEMENTS, \
                f"Invalid camera movement: {scene['camera_movement']}"

        # Check total duration
        total = sum(s["duration"] for s in result["scenes"])
        assert abs(total - 30) <= 3, f"Total duration {total} should be within 3 seconds of target"

        print(f"  PASS: Valid JSON structure and content")
        print(f"    - Scenes generated: {len(result['scenes'])}")
        print(f"    - Total duration: {total}s")
        print(f"    - Camera movements: {set(s['camera_movement'] for s in result['scenes'])}")

        return result

    except Exception as e:
        print(f"  FAIL: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def test_scene_count_scaling():
    """Test Case 2.2: Different Thread Lengths"""
    print("\n" + "="*60)
    print("Test Case 2.2: Scene Count Scaling")
    print("="*60)

    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")

    if not api_key:
        print("  SKIPPED: No API key set")
        return

    provider = "openai" if os.getenv("OPENAI_API_KEY") else "anthropic"

    try:
        scene_gen = SceneGenerator(api_key=api_key, provider=provider)

        test_cases = [
            {
                "name": "short",
                "text": "AI is changing the world. It helps us create. The future is exciting.",
                "expected_range": (6, 7)
            },
            {
                "name": "medium",
                "text": "AI is changing the world. " + "New models emerge daily. " * 5,
                "expected_range": (7, 9)
            }
        ]

        for case in test_cases:
            thread_data = {"thread_id": f"test_{case['name']}", "combined_text": case['text']}

            result = scene_gen.generate_scenes(
                thread_data,
                art_style="minimalist",
                target_duration=30
            )

            scene_count = len(result["scenes"])
            min_expected, max_expected = case["expected_range"]

            if min_expected <= scene_count <= max_expected:
                print(f"  PASS: {case['name']} thread -> {scene_count} scenes")
            else:
                print(f"  WARN: {case['name']} thread -> {scene_count} scenes "
                      f"(expected {min_expected}-{max_expected})")

    except Exception as e:
        print(f"  FAIL: {str(e)}")


def test_camera_movements_valid():
    """Test Case 2.3a: Camera Movements Constants"""
    print("\n" + "="*60)
    print("Test Case 2.3a: Camera Movements Validation")
    print("="*60)

    expected = ["static", "zoom_in", "zoom_out", "pan_left", "pan_right"]

    if VALID_CAMERA_MOVEMENTS == expected:
        print(f"  PASS: Valid camera movements defined correctly")
        print(f"    Movements: {VALID_CAMERA_MOVEMENTS}")
    else:
        print(f"  FAIL: Camera movements mismatch")
        print(f"    Expected: {expected}")
        print(f"    Got: {VALID_CAMERA_MOVEMENTS}")


def test_save_script():
    """Test Case 2.3b: Save Script"""
    print("\n" + "="*60)
    print("Test Case 2.3b: Save Script")
    print("="*60)

    # Mock scene data
    test_data = {
        "metadata": {
            "art_style": "minimalist geometric",
            "total_duration": 30,
            "scene_count": 3
        },
        "scenes": [
            {
                "scene_number": 1,
                "duration": 3.0,
                "visual_description": "A blue circle",
                "narration_text": "First scene",
                "camera_movement": "static"
            },
            {
                "scene_number": 2,
                "duration": 4.0,
                "visual_description": "Circle grows",
                "narration_text": "Second scene",
                "camera_movement": "zoom_in"
            },
            {
                "scene_number": 3,
                "duration": 3.0,
                "visual_description": "Circle fades",
                "narration_text": "Third scene",
                "camera_movement": "zoom_out"
            }
        ]
    }

    output_path = "output/scripts/test_script.json"

    try:
        # Create a dummy scene_gen instance (we just need the save method)
        # We'll mock it since we don't have an API key for this test
        scene_gen = SceneGenerator.__new__(SceneGenerator)
        scene_gen.save_script(test_data, output_path)

        # Verify file exists and contains correct data
        assert os.path.exists(output_path), f"File should exist: {output_path}"

        with open(output_path, 'r') as f:
            loaded_data = json.load(f)

        assert loaded_data == test_data, "Saved data should match input data"

        print(f"  PASS: Script saved and loaded correctly")
        print(f"    Path: {output_path}")

        # Cleanup
        os.remove(output_path)

        return True

    except Exception as e:
        print(f"  FAIL: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_parse_response():
    """Test Case 2.3c: Parse Response"""
    print("\n" + "="*60)
    print("Test Case 2.3c: Parse Response")
    print("="*60)

    # Test different response formats
    test_cases = [
        {
            "name": "Clean JSON",
            "response": '{"metadata": {"scene_count": 1}, "scenes": []}',
            "should_pass": True
        },
        {
            "name": "Markdown code block",
            "response": '```json\n{"metadata": {"scene_count": 1}, "scenes": []}\n```',
            "should_pass": True
        },
        {
            "name": "Text before JSON",
            "response": 'Here is the result: {"metadata": {"scene_count": 1}, "scenes": []}',
            "should_pass": True
        },
        {
            "name": "Invalid JSON",
            "response": 'This is not JSON',
            "should_pass": False
        }
    ]

    scene_gen = SceneGenerator.__new__(SceneGenerator)

    all_passed = True

    for case in test_cases:
        try:
            result = scene_gen._parse_response(case["response"])

            if case["should_pass"]:
                print(f"  PASS: {case['name']}")
            else:
                print(f"  FAIL: {case['name']} - should have raised error")
                all_passed = False

        except json.JSONDecodeError:
            if not case["should_pass"]:
                print(f"  PASS: {case['name']} - correctly raised error")
            else:
                print(f"  FAIL: {case['name']} - unexpected error")
                all_passed = False

    if all_passed:
        print("\n  PASS: All parse tests passed")


def test_validation():
    """Test Case 2.3d: Scene Validation"""
    print("\n" + "="*60)
    print("Test Case 2.3d: Scene Validation")
    print("="*60)

    scene_gen = SceneGenerator.__new__(SceneGenerator)

    # Valid data
    valid_data = {
        "metadata": {
            "art_style": "minimalist",
            "total_duration": 30,
            "scene_count": 6
        },
        "scenes": [
            {
                "scene_number": i + 1,
                "duration": 5.0,
                "visual_description": f"Scene {i+1}",
                "narration_text": f"Narration {i+1}",
                "camera_movement": "static"
            }
            for i in range(6)
        ]
    }

    # Test cases
    test_cases = [
        {
            "name": "Valid data",
            "data": valid_data,
            "should_pass": True
        },
        {
            "name": "Missing metadata",
            "data": {"scenes": []},
            "should_pass": False
        },
        {
            "name": "Too few scenes",
            "data": {
                "metadata": {"art_style": "test", "total_duration": 10, "scene_count": 3},
                "scenes": [
                    {
                        "scene_number": 1,
                        "duration": 3.0,
                        "visual_description": "Test",
                        "narration_text": "Test",
                        "camera_movement": "static"
                    }
                ] * 3
            },
            "should_pass": False
        },
        {
            "name": "Invalid duration",
            "data": {
                "metadata": {"art_style": "test", "total_duration": 30, "scene_count": 6},
                "scenes": [
                    {
                        "scene_number": i + 1,
                        "duration": 1.0,  # Too short
                        "visual_description": "Test",
                        "narration_text": "Test",
                        "camera_movement": "static"
                    }
                    for i in range(6)
                ]
            },
            "should_pass": False
        },
        {
            "name": "Invalid camera movement",
            "data": {
                "metadata": {"art_style": "test", "total_duration": 30, "scene_count": 6},
                "scenes": [
                    {
                        "scene_number": i + 1,
                        "duration": 5.0,
                        "visual_description": "Test",
                        "narration_text": "Test",
                        "camera_movement": "invalid_movement"
                    }
                    for i in range(6)
                ]
            },
            "should_pass": False
        }
    ]

    for case in test_cases:
        try:
            scene_gen._validate_scene_data(case["data"], target_duration=30)

            if case["should_pass"]:
                print(f"  PASS: {case['name']}")
            else:
                print(f"  FAIL: {case['name']} - should have raised error")

        except (ValueError, AssertionError) as e:
            if not case["should_pass"]:
                print(f"  PASS: {case['name']} - correctly raised error")
            else:
                print(f"  FAIL: {case['name']} - unexpected error: {e}")


def run_all_tests():
    """Run all scene generator tests"""
    print("\n" + "="*60)
    print("SCENE GENERATOR - RUNNING ALL TESTS")
    print("="*60)

    # Run tests that don't require API
    test_camera_movements_valid()
    test_save_script()
    test_parse_response()
    test_validation()

    # Run tests that require API
    print("\n" + "="*60)
    print("Note: The following tests require OPENAI_API_KEY or ANTHROPIC_API_KEY")
    print("="*60)
    test_valid_json_output()
    test_scene_count_scaling()

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
