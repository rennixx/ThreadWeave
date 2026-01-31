"""
Thread Scraper Tests

Module-level testing for ThreadScraper class.
"""

import json
import os
import sys
from pathlib import Path

# Set UTF-8 encoding for Windows console
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.scraper import ThreadScraper


def test_valid_thread():
    """Test Case 1.1: Valid Thread URL"""
    print("\n" + "="*60)
    print("Test Case 1.1: Valid Thread URL")
    print("="*60)

    # Load bearer token from environment
    bearer_token = os.getenv("TWITTER_BEARER_TOKEN")

    if not bearer_token:
        print("⚠️  SKIPPED: TWITTER_BEARER_TOKEN not set in environment")
        print("   Set it with: export TWITTER_BEARER_TOKEN='your_token'")
        return None

    # Use a real, public thread URL for testing
    # This is a well-known thread that should be accessible
    test_urls = [
        "https://twitter.com/goodfellow_ian/status/1520084376677688320",  # Example thread
    ]

    scraper = ThreadScraper(bearer_token=bearer_token)

    for thread_url in test_urls:
        print(f"\nTesting URL: {thread_url}")

        try:
            result = scraper.extract_thread(thread_url)

            # Assertions
            assert result is not None, "Should return thread data"
            assert "thread_id" in result, "Should have thread_id"
            assert "tweets" in result, "Should have tweets array"
            assert len(result["tweets"]) >= 2, "Thread should have at least 2 tweets"
            assert "combined_text" in result, "Should have combined text"

            # Check text cleaning
            combined = result["combined_text"]
            assert "http" not in combined, "URLs should be removed"
            assert "@" not in combined or combined.index("@") == 0, "Mentions should be removed (@author may remain)"

            print(f"  ✅ Test 1.1 passed!")
            print(f"     Thread ID: {result['thread_id']}")
            print(f"     Author: {result['author']}")
            print(f"     Tweet count: {result['tweet_count']}")
            print(f"     Combined text length: {len(combined)} chars")

            return result

        except Exception as e:
            print(f"  ❌ Test 1.1 failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return None


def test_invalid_url():
    """Test Case 1.2: Invalid URL"""
    print("\n" + "="*60)
    print("Test Case 1.2: Invalid URL")
    print("="*60)

    scraper = ThreadScraper(bearer_token="dummy_token")

    invalid_urls = [
        "https://twitter.com/invalid",
        "https://example.com/not-twitter",
        "not-a-url",
        "https://twitter.com/user/status/abc",  # Invalid ID
    ]

    for url in invalid_urls:
        print(f"\nTesting invalid URL: {url}")

        try:
            result = scraper.extract_thread(url)
            print(f"  ❌ FAIL: Should raise exception for invalid URL, got: {result}")
        except ValueError as e:
            print(f"  ✅ PASS: Correctly raises ValueError - {str(e)}")
        except Exception as e:
            print(f"  ⚠️  Raises different exception: {type(e).__name__}: {str(e)}")


def test_text_cleaning():
    """Test Case 1.2b: Text Cleaning"""
    print("\n" + "="*60)
    print("Test Case 1.2b: Text Cleaning")
    print("="*60)

    scraper = ThreadScraper()

    test_cases = [
        {
            "input": "Check this out! https://example.com #amazing",
            "expected": "Check this out! amazing",
            "description": "Remove URL, keep hashtag text"
        },
        {
            "input": "Hey @user123, look at this #stuff",
            "expected": "Hey , look at this stuff",
            "description": "Remove mentions"
        },
        {
            "input": "  Extra    whitespace   everywhere  ",
            "expected": "Extra whitespace everywhere",
            "description": "Normalize whitespace"
        },
        {
            "input": "Link https://t.co/abc123 and mention @user",
            "expected": "Link and mention",
            "description": "Remove t.co links and mentions"
        }
    ]

    all_passed = True

    for case in test_cases:
        result = scraper.clean_text(case["input"])
        expected = case["expected"]

        if result == expected:
            print(f"  ✅ {case['description']}")
        else:
            print(f"  ❌ {case['description']}")
            print(f"     Input: '{case['input']}'")
            print(f"     Expected: '{expected}'")
            print(f"     Got: '{result}'")
            all_passed = False

    if all_passed:
        print("\n✅ Test 1.2b passed: All text cleaning tests passed")
    else:
        print("\n⚠️  Test 1.2b: Some tests failed")


def test_url_parsing():
    """Test Case 1.2c: URL Parsing"""
    print("\n" + "="*60)
    print("Test Case 1.2c: URL Parsing")
    print("="*60)

    scraper = ThreadScraper()

    test_cases = [
        {
            "url": "https://twitter.com/user/status/123456789",
            "expected_id": "123456789",
            "expected_user": "user"
        },
        {
            "url": "https://x.com/elonmusk/status/987654321",
            "expected_id": "987654321",
            "expected_user": "elonmusk"
        },
        {
            "url": "https://www.twitter.com/user_name/status/111222333",
            "expected_id": "111222333",
            "expected_user": "user_name"
        },
        {
            "url": "https://example.com/not-twitter",
            "expected_id": None,
            "expected_user": None
        }
    ]

    all_passed = True

    for case in test_cases:
        tweet_id, username = scraper._parse_url(case["url"])

        if tweet_id == case["expected_id"] and username == case["expected_user"]:
            print(f"  ✅ {case['url']}")
        else:
            print(f"  ❌ {case['url']}")
            print(f"     Expected: ({case['expected_id']}, {case['expected_user']})")
            print(f"     Got: ({tweet_id}, {username})")
            all_passed = False

    if all_passed:
        print("\n✅ Test 1.2c passed: All URL parsing tests passed")
    else:
        print("\n⚠️  Test 1.2c: Some tests failed")


def test_save_thread():
    """Test Case 1.2d: Save Thread"""
    print("\n" + "="*60)
    print("Test Case 1.2d: Save Thread")
    print("="*60)

    scraper = ThreadScraper()

    # Mock thread data
    test_data = {
        "thread_id": "test123",
        "author": "@testuser",
        "created_at": "2024-01-01T00:00:00Z",
        "tweet_count": 3,
        "tweets": [
            {"order": 1, "id": "1", "text": "First tweet", "original_text": "First tweet"},
            {"order": 2, "id": "2", "text": "Second tweet", "original_text": "Second tweet"},
            {"order": 3, "id": "3", "text": "Third tweet", "original_text": "Third tweet"}
        ],
        "combined_text": "First tweet\n\nSecond tweet\n\nThird tweet"
    }

    output_path = "output/threads/test_thread.json"

    try:
        # Create output directory if needed
        os.makedirs("output/threads", exist_ok=True)

        scraper.save_thread(test_data, output_path)

        # Verify file exists and contains correct data
        assert os.path.exists(output_path), f"File should exist: {output_path}"

        with open(output_path, 'r') as f:
            loaded_data = json.load(f)

        assert loaded_data == test_data, "Saved data should match input data"

        print(f"  ✅ Test 1.2d passed!")
        print(f"     Thread data saved to: {output_path}")

        # Cleanup
        os.remove(output_path)

        return True

    except Exception as e:
        print(f"  ❌ Test 1.2d failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all scraper tests"""
    print("\n" + "="*60)
    print("THREAD SCRAPER - RUNNING ALL TESTS")
    print("="*60)

    # Run tests that don't require API
    test_url_parsing()
    test_text_cleaning()
    test_save_thread()

    # Run tests that require API (will skip if no token)
    print("\n" + "="*60)
    print("Note: The following test requires TWITTER_BEARER_TOKEN")
    print("="*60)
    test_valid_thread()

    print("\n" + "="*60)
    print("INVALID URL TEST (no API required)")
    print("="*60)
    test_invalid_url()

    print("\n" + "="*60)
    print("ALL TESTS COMPLETED")
    print("="*60)


if __name__ == "__main__":
    run_all_tests()
