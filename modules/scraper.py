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

import requests
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

        # Try API if available, fallback to web scraping on failure
        thread_data = None
        if self.client:
            try:
                thread_data = self._extract_via_api(thread_id)
            except Exception as e:
                logger.warning(f"API extraction failed: {e}")
                logger.info("Falling back to web scraping...")
                thread_data = None

        # Fallback to web scraping
        if not thread_data:
            thread_data = self._extract_via_web(thread_url)

        # Validate result
        if not thread_data or thread_data.get("tweet_count", 0) < 1:
            raise ValueError(
                f"No valid tweets found in thread."
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

            except tweepy.errors.TooManyRequests as e:
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)
                    logger.warning(f"Rate limited. Waiting {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    raise RuntimeError("Rate limit exceeded. Please try again later.")

            except tweepy.errors.Forbidden as e:
                raise RuntimeError(f"Access forbidden. Check API credentials: {e}")

            except tweepy.errors.NotFound:
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

        Uses multiple approaches: Nitter instances, and direct parsing.

        Args:
            url: Thread URL

        Returns:
            dict: Thread data

        Raises:
            RuntimeError: If scraping fails
        """
        from bs4 import BeautifulSoup

        # List of public Nitter instances
        nitter_instances = [
            "nitter.net",
            "nitter.poast.org",
            "nitter.privacydev.net",
            "nitter.1d4.us",
            "nitter.kavin.rocks",
        ]

        # Parse original URL to get tweet ID and username
        thread_id, username = self._parse_url(url)

        # Try multiple Nitter instances
        for instance in nitter_instances:
            try:
                # Build Nitter URL
                nitter_url = f"https://{instance}/{username}/status/{thread_id}"

                logger.info(f"Trying Nitter instance: {instance}")

                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                    "Accept-Language": "en-US,en;q=0.5",
                }

                response = requests.get(nitter_url, headers=headers, timeout=15)

                if response.status_code != 200:
                    logger.warning(f"Instance {instance} returned {response.status_code}")
                    continue

                # Parse HTML with more flexible encoding handling
                soup = BeautifulSoup(response.content, 'html.parser')

                # Extract thread content - try multiple selectors
                tweets = []

                # Method 1: Try timeline-item divs
                for item in soup.find_all('div', class_='timeline-item'):
                    tweet_content = item.find('div', class_='tweet-content')
                    if tweet_content:
                        text = tweet_content.get_text(separator=' ', strip=True)
                        if text and len(text) > 20:
                            tweets.append({"text": text, "order": len(tweets) + 1})

                # Method 2: Try tweet divs
                if not tweets:
                    for item in soup.find_all('div', class_='tweet'):
                        tweet_content = item.find('div', class_='tweet-content')
                        if tweet_content:
                            text = tweet_content.get_text(separator=' ', strip=True)
                            if text and len(text) > 20:
                                tweets.append({"text": text, "order": len(tweets) + 1})

                # Method 3: Try finding main-content divs
                if not tweets:
                    for item in soup.find_all('div', class_='main-tweet'):
                        text = item.get_text(separator=' ', strip=True)
                        if text and len(text) > 20:
                            tweets.append({"text": text, "order": len(tweets) + 1})

                # Method 4: Look for any div with class containing 'content' or 'text'
                if not tweets:
                    for item in soup.find_all('div', class_=lambda x: x and ('content' in x.lower() or 'text' in x.lower())):
                        text = item.get_text(separator=' ', strip=True)
                        if text and 50 < len(text) < 1000:  # Reasonable tweet length
                            tweets.append({"text": text, "order": len(tweets) + 1})

                # Method 5: Last resort - get all paragraph tags
                if not tweets:
                    all_text = []
                    for p in soup.find_all('p'):
                        text = p.get_text(strip=True)
                        if text and len(text) > 20:
                            all_text.append(text)
                    if all_text:
                        # Take first few paragraphs as they're likely the tweet
                        for i, text in enumerate(all_text[:5]):
                            tweets.append({"text": text, "order": i + 1})

                if not tweets:
                    continue

                # Clean and combine text
                cleaned_tweets = []
                for t in tweets:
                    cleaned = self.clean_text(t["text"])
                    if cleaned and len(cleaned) > 10:
                        # Remove duplicates
                        if cleaned not in cleaned_tweets:
                            cleaned_tweets.append(cleaned)

                if len(cleaned_tweets) < 1:
                    continue

                combined_text = "\n\n".join(cleaned_tweets)

                logger.info(f"Successfully scraped {len(cleaned_tweets)} tweets from {instance}")

                return {
                    "thread_id": thread_id,
                    "author": f"@{username}",
                    "created_at": None,
                    "tweet_count": len(cleaned_tweets),
                    "tweets": [
                        {
                            "order": i + 1,
                            "id": f"{thread_id}_{i + 1}",
                            "text": cleaned_tweets[i],
                            "original_text": cleaned_tweets[i]
                        }
                        for i in range(len(cleaned_tweets))
                    ],
                    "combined_text": combined_text
                }

            except requests.exceptions.RequestException as e:
                logger.warning(f"Instance {instance} network error: {e}")
                continue
            except Exception as e:
                logger.warning(f"Instance {instance} failed: {e}")
                continue

        raise RuntimeError(
            "All scraping methods failed. The thread may be private, deleted, "
            "or the scraping service is temporarily down. Options:\n"
            "1. Try again later\n"
            "2. Use Twitter API (requires paid tier)\n"
            "3. Provide thread text manually"
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
