"""
Thread Scraper Module

Extracts Twitter/X thread data and converts to structured format.
"""

import json
import logging
import re
import time
from typing import Optional
from urllib.parse import parse_qs, urlparse

import tweepy
from tqdm import tqdm

logger = logging.getLogger(__name__)


class ThreadScraper:
    """
    Scrapes Twitter/X threads and converts them to structured format.

    Supports both API-based scraping (preferred) and web scraping fallback.
    """

    # Twitter URL patterns
    TWEET_URL_PATTERN = re.compile(
        r'https?://(www\.)?(twitter|x)\.com/([a-zA-Z0-9_]+)/status/([0-9]+)'
    )

    def __init__(self, bearer_token: Optional[str] = None):
        """
        Initialize the scraper.

        Args:
            bearer_token: Twitter API v2 bearer token.
                         If None, web scraping fallback will be used.
        """
        self.bearer_token = bearer_token
        self.client = None

        if bearer_token:
            try:
                self.client = tweepy.Client(bearer_token=bearer_token)
                logger.info("ThreadScraper initialized with Twitter API")
            except Exception as e:
                logger.warning(f"Failed to initialize Twitter API client: {e}")
                logger.info("Will attempt web scraping fallback")

    def extract_thread(self, thread_url: str) -> dict:
        """
        Extract a Twitter thread from its URL.

        Args:
            thread_url: URL of a tweet in the thread

        Returns:
            dict: Structured thread data with keys:
                - thread_id: Unique identifier
                - author: Username of thread author
                - created_at: ISO timestamp
                - tweet_count: Number of tweets
                - tweets: List of tweet objects
                - combined_text: All tweets combined

        Raises:
            ValueError: If URL is invalid
            RuntimeError: If scraping fails
        """
        # Validate URL
        thread_id, username = self._parse_url(thread_url)

        if not thread_id:
            raise ValueError(f"Invalid thread URL: {thread_url}")

        logger.info(f"Extracting thread: {thread_id} by @{username}")

        # Use API if available, otherwise fallback
        if self.client:
            thread_data = self._extract_via_api(thread_id)
        else:
            thread_data = self._extract_via_web(thread_url)

        # Validate result
        if not thread_data or thread_data.get("tweet_count", 0) < 2:
            raise ValueError(
                f"Thread must have at least 2 tweets. "
                f"Found: {thread_data.get('tweet_count', 0)}"
            )

        logger.info(
            f"Successfully extracted {thread_data['tweet_count']} tweets "
            f"from @{thread_data['author']}"
        )

        return thread_data

    def _parse_url(self, url: str) -> tuple[Optional[str], Optional[str]]:
        """
        Parse a Twitter URL to extract tweet ID and username.

        Args:
            url: Twitter status URL

        Returns:
            tuple: (tweet_id, username) or (None, None) if invalid
        """
        match = self.TWEET_URL_PATTERN.match(url)

        if match:
            return match.group(4), match.group(3)

        return None, None

    def _extract_via_api(self, tweet_id: str) -> dict:
        """
        Extract thread using Twitter API v2.

        Args:
            tweet_id: ID of the starting tweet

        Returns:
            dict: Thread data

        Raises:
            RuntimeError: If API call fails
        """
        max_retries = 3
        retry_delay = 5  # seconds

        for attempt in range(max_retries):
            try:
                # Get the original tweet
                tweet = self.client.get_tweet(
                    tweet_id,
                    expansions=["author_id", "in_reply_to_user_id"],
                    tweet_fields=["created_at", "conversation_id", "text", "author_id"],
                    user_fields=["username"]
                )

                if not tweet.data:
                    raise RuntimeError(f"Tweet {tweet_id} not found (may be deleted)")

                # Get conversation ID
                conversation_id = tweet.data.conversation_id
                author_id = tweet.data.author_id

                # Get all tweets in the conversation
                timeline = self.client.search_recent_tweets(
                    query=f"conversation_id:{conversation_id}",
                    expansions=["author_id"],
                    tweet_fields=["created_at", "text", "author_id"],
                    user_fields=["username"],
                    max_results=100
                )

                if not timeline.data:
                    raise RuntimeError("No tweets found in conversation")

                # Get user info
                users = {u.id: u.username for u in timeline.includes.get("users", [])}
                author_username = users.get(author_id, "unknown")

                # Filter for thread author's tweets only
                thread_tweets = []
                for t in timeline.data:
                    # Check if this tweet is from the thread author
                    tweet_author = users.get(t.author_id, "")
                    if tweet_author == author_username:
                        thread_tweets.append({
                            "id": t.id,
                            "text": t.text,
                            "created_at": t.created_at.isoformat(),
                            "order": len(thread_tweets) + 1
                        })

                # Sort by creation time
                thread_tweets.sort(key=lambda x: x["created_at"])

                # Build combined text
                combined_text = "\n\n".join([
                    self.clean_text(t["text"]) for t in thread_tweets
                ])

                return {
                    "thread_id": conversation_id,
                    "author": f"@{author_username}",
                    "created_at": thread_tweets[0]["created_at"] if thread_tweets else None,
                    "tweet_count": len(thread_tweets),
                    "tweets": [
                        {
                            "order": t["order"],
                            "id": t["id"],
                            "text": self.clean_text(t["text"]),
                            "original_text": t["text"]
                        }
                        for t in thread_tweets
                    ],
                    "combined_text": combined_text
                }

            except tweepy.Errors.TooManyRequests as e:
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)
                    logger.warning(f"Rate limited. Waiting {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    raise RuntimeError("Rate limit exceeded. Please try again later.")

            except tweepy.Errors.Forbidden as e:
                raise RuntimeError(f"Access forbidden. Check API credentials: {e}")

            except tweepy.Errors.NotFound:
                raise RuntimeError(f"Tweet {tweet_id} not found (may be deleted or private)")

            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Attempt {attempt + 1} failed: {e}")
                    time.sleep(retry_delay)
                else:
                    raise RuntimeError(f"Failed to extract thread after {max_retries} attempts: {e}")

        raise RuntimeError("Failed to extract thread")

    def _extract_via_web(self, url: str) -> dict:
        """
        Extract thread using web scraping (fallback method).

        Note: This method is fragile and may break without warning.
        Consider using the official API.

        Args:
            url: Thread URL

        Returns:
            dict: Thread data

        Raises:
            NotImplementedError: Web scraping not yet implemented
        """
        # TODO: Implement web scraping fallback using requests + beautifulsoup4
        # This is complex due to Twitter's dynamic content loading
        raise NotImplementedError(
            "Web scraping fallback not implemented. "
            "Please provide a Twitter API bearer token."
        )

    def clean_text(self, text: str) -> str:
        """
        Clean tweet text by removing URLs, mentions, and hashtags.

        Args:
            text: Raw tweet text

        Returns:
            str: Cleaned text
        """
        # Remove URLs
        text = re.sub(r'https?://\S+', '', text)

        # Remove mentions (@username)
        text = re.sub(r'@\w+', '', text)

        # Remove hashtags (keep the text, remove the #)
        text = re.sub(r'#(\w+)', r'\1', text)

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()

        return text

    def save_thread(self, thread_data: dict, output_path: str) -> None:
        """
        Save thread data to a JSON file.

        Args:
            thread_data: Thread data from extract_thread()
            output_path: Path to save the JSON file
        """
        output_path = str(output_path)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(thread_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Thread data saved to: {output_path}")
