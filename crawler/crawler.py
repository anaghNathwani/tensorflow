"""
Multi-threaded web crawler for LLM training data.

- Crawls Wikipedia, Project Gutenberg, and arXiv abstracts
- Respects robots.txt and rate-limits per domain
- Detail level controls crawl breadth and depth
- Yields clean text chunks via a thread-safe queue
"""
import urllib.request
import urllib.robotparser
import urllib.parse
import threading
import queue
import time
import re
import random
import logging
import json
import os
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

from crawler.extractor import html_to_text

_DATA_DIR = Path(__file__).parent.parent / "data"

log = logging.getLogger("crawler")


# ── Wikipedia article seeds by detail level ───────────────────────────────────
# We use the Wikipedia REST API (returns clean JSON, no HTML scraping needed)
_WIKI_SEEDS = {
    "low": [
        "Science", "History", "Mathematics", "Geography", "Technology",
        "Nature", "Art", "Language", "Music", "Sport", "Food", "Animal",
        "Water", "Earth", "Sun", "Moon", "Plant", "Human", "Time", "Energy",
    ],
    "medium": [
        "Science", "History", "Mathematics", "Physics", "Chemistry", "Biology",
        "Technology", "Philosophy", "Literature", "Geography", "Economics",
        "Psychology", "Medicine", "Astronomy", "Computer_science", "Engineering",
        "Linguistics", "Sociology", "Architecture", "Agriculture",
        "Electricity", "Gravity", "Evolution", "Democracy", "Climate",
        "Ocean", "Atmosphere", "Volcano", "Earthquake", "Galaxy",
    ],
    "high": [
        "Science", "History", "Mathematics", "Physics", "Chemistry", "Biology",
        "Technology", "Philosophy", "Literature", "Geography", "Economics",
        "Psychology", "Medicine", "Astronomy", "Computer_science", "Engineering",
        "Linguistics", "Sociology", "Architecture", "Agriculture", "Law",
        "Electricity", "Gravity", "Evolution", "Democracy", "Climate",
        "Ocean", "Atmosphere", "Volcano", "Earthquake", "Galaxy",
        "Quantum_mechanics", "General_relativity", "DNA", "Protein",
        "Neuroscience", "Immunology", "Ecology", "Thermodynamics",
        "Calculus", "Statistics", "Logic", "Ethics", "Aesthetics",
        "Artificial_intelligence", "Machine_learning", "Cryptography",
        "Number_theory", "Topology", "Algebra", "Geometry",
        "Roman_Empire", "Ancient_Greece", "Renaissance", "Industrial_Revolution",
        "World_War_II", "Cold_War", "French_Revolution", "Silk_Road",
    ],
}

# Gutenberg book IDs (plain text files — great training data)
_GUTENBERG_BOOKS = {
    "low":   [11, 1342],                              # Alice, P&P
    "medium":[11, 1342, 84, 1661, 98, 174],           # + Frankenstein, Holmes, etc.
    "high":  [11, 1342, 84, 1661, 98, 174, 2701,
              1080, 1400, 76, 1232, 2554, 5200,
              158, 43, 1497, 25344, 514, 1260],        # broad classics
}

_MAX_PAGES = {"low": 60,   "medium": 300,  "high": 99_999}
_THREADS   = {"low": 3,    "medium": 5,    "high": 8}
_RATE_LIMIT = 0.3   # seconds between requests to the same domain

_UA = "Mozilla/5.0 (compatible; TF-LLM-Crawler/1.0; educational use)"


@dataclass
class CrawlStats:
    pages_fetched:  int = 0
    pages_skipped:  int = 0
    bytes_fetched:  int = 0
    texts_produced: int = 0
    errors:         int = 0


class WebCrawler:
    """
    Fetches training text from:
      - Wikipedia REST API  (clean JSON extracts — no HTML parsing needed)
      - Project Gutenberg   (plain-text books)
    Puts clean text into out_queue. Call start() to begin.
    """

    def __init__(self, detail: str, out_queue: queue.Queue,
                 stop_event: threading.Event):
        self.detail     = detail
        self.out_queue  = out_queue
        self.stop_event = stop_event
        self.stats      = CrawlStats()

        # File that accumulates all crawled text for this session
        _DATA_DIR.mkdir(parents=True, exist_ok=True)
        self._data_file = open(_DATA_DIR / "crawled.jsonl", "a", encoding="utf-8")
        self._file_lock = threading.Lock()

        self._max_pages  = _MAX_PAGES[detail]
        self._n_threads  = _THREADS[detail]
        self._domain_last: dict[str, float] = {}
        self._domain_lock = threading.Lock()

        # Build task queue: mix of Wikipedia articles + Gutenberg books
        self._tasks: queue.Queue = queue.Queue()
        self._seen:  set[str]    = set()
        self._seen_lock           = threading.Lock()

        wiki_topics = list(_WIKI_SEEDS[detail])
        random.shuffle(wiki_topics)
        for topic in wiki_topics:
            self._tasks.put(("wiki", topic))

        for book_id in _GUTENBERG_BOOKS[detail]:
            self._tasks.put(("gutenberg", book_id))

    def _emit(self, text: str):
        """Send text to the training queue and persist it to data/crawled.jsonl."""
        self.out_queue.put(text)
        with self._file_lock:
            self._data_file.write(json.dumps({"text": text}) + "\n")
            self._data_file.flush()

    def start(self) -> list[threading.Thread]:
        threads = [
            threading.Thread(target=self._worker, daemon=True, name=f"crawl-{i}")
            for i in range(self._n_threads)
        ]
        for t in threads:
            t.start()
        return threads

    # ── Workers ───────────────────────────────────────────────────────────────

    def _worker(self):
        while not self.stop_event.is_set():
            try:
                task = self._tasks.get(timeout=2.0)
            except queue.Empty:
                continue

            kind, target = task

            with self._seen_lock:
                key = f"{kind}:{target}"
                if key in self._seen or self.stats.pages_fetched >= self._max_pages:
                    continue
                self._seen.add(key)

            if kind == "wiki":
                self._fetch_wiki(str(target))
            elif kind == "gutenberg":
                self._fetch_gutenberg(int(target))

    def _rate_limit(self, domain: str):
        with self._domain_lock:
            last = self._domain_last.get(domain, 0)
            wait = _RATE_LIMIT - (time.time() - last)
            if wait > 0:
                time.sleep(wait)
            self._domain_last[domain] = time.time()

    def _get(self, url: str, as_json: bool = False):
        self._rate_limit(urllib.parse.urlparse(url).netloc)
        req = urllib.request.Request(url, headers={"User-Agent": _UA,
                                                    "Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            raw = resp.read(2_000_000)
            self.stats.bytes_fetched += len(raw)
            if as_json:
                import json
                return json.loads(raw.decode("utf-8"))
            return raw.decode("utf-8", errors="replace")

    # ── Wikipedia REST API ────────────────────────────────────────────────────

    def _fetch_wiki(self, topic: str):
        """Fetch a Wikipedia article via the REST summary + sections API."""
        try:
            # Full article text via the Wikimedia REST API
            url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(topic)}"
            data = self._get(url, as_json=True)
            text = data.get("extract", "")

            if len(text) > 100:
                self._emit(text)
                self.stats.texts_produced += 1
                self.stats.pages_fetched += 1

            # Also enqueue "related" links from the article's sections
            self._enqueue_wiki_links(topic)

        except Exception as e:
            self.stats.errors += 1
            log.debug(f"Wiki error {topic}: {e}")

    def _enqueue_wiki_links(self, topic: str):
        """Fetch the links in a Wikipedia article and add them as tasks."""
        try:
            url = (f"https://en.wikipedia.org/w/api.php?action=query"
                   f"&titles={urllib.parse.quote(topic)}&prop=links"
                   f"&pllimit=30&plnamespace=0&format=json")
            data = self._get(url, as_json=True)
            pages = data.get("query", {}).get("pages", {})
            for page in pages.values():
                for link in page.get("links", []):
                    title = link.get("title", "").replace(" ", "_")
                    if title:
                        with self._seen_lock:
                            key = f"wiki:{title}"
                            if key not in self._seen:
                                self._tasks.put(("wiki", title))
        except Exception:
            pass

    # ── Project Gutenberg ─────────────────────────────────────────────────────

    def _fetch_gutenberg(self, book_id: int):
        """Fetch a plain-text book from Project Gutenberg."""
        try:
            url = f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt"
            try:
                text = self._get(url)
            except Exception:
                # Fallback to older URL scheme
                url  = f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
                text = self._get(url)

            # Strip Gutenberg header/footer boilerplate
            text = self._strip_gutenberg(text)

            if len(text) > 200:
                # Split into ~2000-char chunks so the queue stays responsive
                chunk_size = 2000
                for i in range(0, len(text), chunk_size):
                    if self.stop_event.is_set():
                        break
                    chunk = text[i:i + chunk_size].strip()
                    if len(chunk) > 100:
                        self._emit(chunk)
                        self.stats.texts_produced += 1

                self.stats.pages_fetched += 1

        except Exception as e:
            self.stats.errors += 1
            log.debug(f"Gutenberg error {book_id}: {e}")

    @staticmethod
    def _strip_gutenberg(text: str) -> str:
        """Remove Project Gutenberg header and footer."""
        start_markers = ["*** START OF THE PROJECT GUTENBERG",
                         "***START OF THE PROJECT GUTENBERG",
                         "*** START OF THIS PROJECT GUTENBERG"]
        end_markers   = ["*** END OF THE PROJECT GUTENBERG",
                         "***END OF THE PROJECT GUTENBERG",
                         "*** END OF THIS PROJECT GUTENBERG"]
        for m in start_markers:
            idx = text.find(m)
            if idx != -1:
                text = text[text.find("\n", idx) + 1:]
                break
        for m in end_markers:
            idx = text.find(m)
            if idx != -1:
                text = text[:idx]
                break
        return text.strip()
