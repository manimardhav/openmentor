"""
Groq API client wrapper.

Why this file exists:
Every other file in this project should call `call_groq()` instead of
importing Groq directly. This means:
  1. Retry/timeout logic lives in exactly one place.
  2. If the team ever switches LLM providers, only this file changes.
  3. Errors are converted into one predictable exception type
     (LLMCallError) instead of leaking raw httpx/groq internals into
     code that shouldn't need to know about them.
"""

import time
import logging
from groq import Groq
from groq import APIConnectionError, APIStatusError

from config import GROQ_API_KEY, MODEL_NAME, MAX_RETRIES, RETRY_BACKOFF_SECONDS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("groq_client")

_client = Groq(api_key=GROQ_API_KEY)


class LLMCallError(Exception):
    """Raised when a Groq call fails after all retries are exhausted."""
    pass


def call_groq(messages: list[dict], temperature: float, max_tokens: int = 600) -> str:
    """
    Sends a chat completion request to Groq with automatic retries.

    Why retries matter here specifically: student wifi and mobile
    hotspots drop connections mid-request (this happened during actual
    development of this project). Rather than letting the whole script
    crash on a transient network blip, we retry with exponential
    backoff (2s, 4s, 8s) before giving up.

    messages: the [{"role": ..., "content": ...}] list built by a
              prompt template function.
    temperature: passed through from the caller — different tasks need
                 different temperatures (see config.py).
    max_tokens: caps response length so a runaway response can't burn
                through API quota silently.
    """
    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = _client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content

        except APIConnectionError as e:
            # Network-level failure (DNS, dropped connection, timeout).
            last_error = e
            wait = RETRY_BACKOFF_SECONDS * (2 ** (attempt - 1))
            logger.warning(
                f"Groq connection error on attempt {attempt}/{MAX_RETRIES}: {e}. "
                f"Retrying in {wait}s..."
            )
            if attempt < MAX_RETRIES:
                time.sleep(wait)

        except APIStatusError as e:
            # Groq responded but with an error status (bad key, rate
            # limit, server error). Rate limits (429) and server errors
            # (5xx) are worth retrying; auth errors (401/403) are not.
            last_error = e
            if e.status_code in (429, 500, 502, 503):
                wait = RETRY_BACKOFF_SECONDS * (2 ** (attempt - 1))
                logger.warning(
                    f"Groq returned {e.status_code} on attempt {attempt}/{MAX_RETRIES}. "
                    f"Retrying in {wait}s..."
                )
                if attempt < MAX_RETRIES:
                    time.sleep(wait)
            else:
                raise LLMCallError(
                    f"Groq call failed with non-retryable status {e.status_code}: {e}"
                ) from e

    raise LLMCallError(
        f"Groq call failed after {MAX_RETRIES} attempts. Last error: {last_error}"
    )