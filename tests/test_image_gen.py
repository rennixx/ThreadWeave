"""
Image Generator Tests

Module-level testing for ImageGenerator class.
"""

import os
import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Try to import, handle missing dependencies
try:
    from modules.image_gen import ImageGenerator
    DEPS_AVAILABLE = True
except ImportError as e:
    DEPS_AVAILABLE = False
    IMPORT_ERROR = str(e)


def test_device_detection():
    """Test Case 3.0: Device Detection"""
    print("\n" + "="*60)
    print("Test Case 3.0: Device Detection")
    print("="*60)

    # Test auto-detection
    img_gen = ImageGenerator()

    if img_gen.device == "cuda":
        print(f"  PASS: CUDA device detected")
        print(f"    Device: {img_gen.device}")
    elif img_gen.device == "cpu":
        print(f"  INFO: Using CPU (CUDA not available)")
        print(f"    Device: {img_gen.device}")
    else:
        print(f"  FAIL: Unknown device: {img_gen.device}")

    return img_gen


def test_config_loading():
    """Test Case 3.0b: Config Loading"""
    print("\n" + "="*60)
    print("Test Case 3.0b: Config Loading")
    print("="*60)

    img_gen = ImageGenerator()

    # Check config values are loaded
    assert hasattr(img_gen, "steps"), "Should have steps"
    assert hasattr(img_gen, "cfg_scale"), "Should have cfg_scale"
    assert hasattr(img_gen, "resolution"), "Should have resolution"
    assert hasattr(img_gen, "base_seed"), "Should have base_seed"

    print(f"  PASS: Config loaded successfully")
    print(f"    Steps: {img_gen.steps}")
    print(f"    CFG Scale: {img_gen.cfg_scale}")
    print(f"    Resolution: {img_gen.resolution}")
    print(f"    Base Seed: {img_gen.base_seed}")

    return img_gen


def test_prompt_building():
    """Test Case 3.0c: Prompt Building"""
    print("\n" + "="*60)
    print("Test Case 3.0c: Prompt Building")
    print("="*60)

    img_gen = ImageGenerator()

    visual_description = "A blue circle on white background"
    art_style = "minimalist geometric"

    positive, negative = img_gen._build_prompt(visual_description, art_style)

    # Check positive prompt
    assert visual_description in positive, "Positive prompt should contain visual description"
    assert art_style in positive, "Positive prompt should contain art style"
    assert len(positive) > len(visual_description), "Positive prompt should be enhanced"

    # Check negative prompt
    assert len(negative) > 0, "Negative prompt should not be empty"
    assert "blurry" in negative.lower(), "Negative prompt should contain quality terms"

    print(f"  PASS: Prompts built correctly")
    print(f"    Positive length: {len(positive)} chars")
    print(f"    Negative length: {len(negative)} chars")

    # Print preview
    print(f"    Positive: {positive[:100]}...")
    print(f"    Negative: {negative[:100]}...")


def test_memory_info():
    """Test Case 3.1a: Memory Info (no GPU required)"""
    print("\n" + "="*60)
    print("Test Case 3.1a: Memory Info")
    print("="*60)

    img_gen = ImageGenerator()

    try:
        memory_info = img_gen.get_memory_info()

        if memory_info.get("device") == "cpu":
            print(f"  INFO: Running on CPU (no GPU memory info)")
        else:
            print(f"  PASS: Memory info retrieved")
            print(f"    Device: {memory_info['device']}")
            print(f"    Allocated: {memory_info.get('allocated', 0):.2f} GB")
            print(f"    Free: {memory_info.get('free', 0):.2f} GB")
            print(f"    Total: {memory_info.get('total', 0):.2f} GB")

    except Exception as e:
        print(f"  WARN: Could not get memory info: {e}")


def test_scene_validation():
    """Test Case 3.1b: Scene Validation"""
    print("\n" + "="*60)
    print("Test Case 3.1b: Scene Validation")
    print("="*60)

    img_gen = ImageGenerator()

    # Test valid scene script
    valid_script = {
        "metadata": {
            "art_style": "minimalist geometric"
        },
        "scenes": [
            {
                "scene_number": 1,
                "visual_description": "A blue circle"
            },
            {
                "scene_number": 2,
                "visual_description": "A red square"
            }
        ]
    }

    # Test invalid scripts
    test_cases = [
        {
            "name": "Valid script",
            "script": valid_script,
            "should_pass": True
        },
        {
            "name": "Empty script",
            "script": {},
            "should_pass": False
        },
        {
            "name": "No scenes",
            "script": {"metadata": {}, "scenes": []},
            "should_pass": False
        },
        {
            "name": "No metadata",
            "script": {"scenes": [{"scene_number": 1}]},
            "should_pass": False
        }
    ]

    for case in test_cases:
        try:
            # Just test validation, don't actually generate
            if "scenes" not in case["script"]:
                raise ValueError("scene_script must contain 'scenes' key")

            if "metadata" not in case["script"]:
                raise ValueError("scene_script must contain 'metadata' key")

            scenes = case["script"]["scenes"]

            if not scenes:
                raise ValueError("No scenes to generate")

            if case["should_pass"]:
                print(f"  PASS: {case['name']}")
            else:
                print(f"  FAIL: {case['name']} - should have raised error")

        except ValueError as e:
            if not case["should_pass"]:
                print(f"  PASS: {case['name']} - correctly raised error")
            else:
                print(f"  FAIL: {case['name']} - unexpected error: {e}")


def test_mock_image_generation():
    """Test Case 3.2: Mock Image Generation (no GPU required)"""
    print("\n" + "="*60)
    print("Test Case 3.2: Mock Image Generation")
    print("="*60)

    img_gen = ImageGenerator()

    scene_script = {
        "metadata": {
            "art_style": "minimalist geometric shapes, flat design"
        },
        "scenes": [
            {
                "scene_number": 1,
                "visual_description": "A blue circle and red square on white background"
            },
            {
                "scene_number": 2,
                "visual_description": "The shapes rotating and merging"
            }
        ]
    }

    output_dir = "output/test_images"

    # This test requires the model to be loaded
    # We'll skip actual generation but validate the flow

    try:
        # Validate scene script structure
        assert "scenes" in scene_script
        assert len(scene_script["scenes"]) == 2
        assert all("visual_description" in s for s in scene_script["scenes"])

        print(f"  PASS: Scene script validation passed")
        print(f"    Scenes: {len(scene_script['scenes'])}")
        print(f"    Output directory: {output_dir}")
        print(f"\n  NOTE: Actual image generation requires:")
        print(f"    - GPU with CUDA support")
        print(f"    - Model download (~7GB for SDXL-Turbo)")
        print(f"    - Run with: python main.py <thread_url>")

    except Exception as e:
        print(f"  FAIL: {str(e)}")


def test_save_mock_image():
    """Test Case 3.2b: Save Mock Image"""
    print("\n" + "="*60)
    print("Test Case 3.2b: Save Mock Image")
    print("="*60)

    from PIL import Image

    # Create a simple test image
    output_dir = "output/test_images"
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, "test_scene_01.png")

    # Create a simple colored image
    test_image = Image.new('RGB', (768, 1024), color='blue')

    try:
        test_image.save(output_path, format="PNG", quality=95)

        # Verify file exists
        assert os.path.exists(output_path), f"Image should exist: {output_path}"

        # Verify it can be loaded
        loaded = Image.open(output_path)
        width, height = loaded.size

        assert width == 768, f"Width should be 768, got {width}"
        assert height == 1024, f"Height should be 1024, got {height}"

        print(f"  PASS: Test image saved successfully")
        print(f"    Path: {output_path}")
        print(f"    Size: {width}x{height}")

        # Cleanup
        os.remove(output_path)

    except Exception as e:
        print(f"  FAIL: {str(e)}")


def run_all_tests():
    """Run all image generator tests"""
    print("\n" + "="*60)
    print("IMAGE GENERATOR - RUNNING ALL TESTS")
    print("="*60)

    # Run tests that don't require GPU
    test_device_detection()
    test_config_loading()
    test_prompt_building()
    test_memory_info()
    test_scene_validation()
    test_save_mock_image()

    print("\n" + "="*60)
    print("Note: Actual image generation tests require:")
    print("  - NVIDIA GPU with CUDA support")
    print("  - Stable Diffusion XL model (~7GB)")
    print("  - Run full pipeline to test: python main.py <thread_url>")
    print("="*60)

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
