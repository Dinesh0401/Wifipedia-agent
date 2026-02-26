# =============================================================================
# llmcontrols_client.py  -- Async LLMControls API Client
# Replaces LiteLLM / OpenAI everywhere in the pipeline
# =============================================================================

import aiohttp
import asyncio
import json
import time
import logging
from typing import Optional

from scripts.config import cfg

logger = logging.getLogger(__name__)


class LLMControlsClient:
    """
    Async client for the LLMControls API.
    Usage:
        client = LLMControlsClient()
        response = await client.generate("Your prompt here")
    """

    def __init__(self, max_retries: int = 3, retry_delay: float = 2.0):
        self.api_key = cfg.llmc_api_key
        self.api_url = cfg.llmc_api_url
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    # -- Core async generate -------------------------------------------------
    async def generate(self, prompt: str, timeout: int = 60) -> str:
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
        }
        payload = {
            "input_value": prompt,
            "output_type": "chat",
            "input_type": "chat",
        }

        for attempt in range(1, self.max_retries + 1):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        self.api_url,
                        json=payload,
                        headers=headers,
                        timeout=aiohttp.ClientTimeout(total=timeout),
                    ) as resp:
                        resp.raise_for_status()
                        data = await resp.json()
                        return self._extract_message(data)
            except aiohttp.ClientResponseError as e:
                logger.warning(f"[Attempt {attempt}] HTTP error: {e.status} {e.message}")
            except asyncio.TimeoutError:
                logger.warning(f"[Attempt {attempt}] Request timed out")
            except Exception as e:
                logger.warning(f"[Attempt {attempt}] Unexpected error: {e}")

            if attempt < self.max_retries:
                await asyncio.sleep(self.retry_delay * attempt)

        raise RuntimeError(f"LLMControls API failed after {self.max_retries} attempts")

    # -- Batch async generate ------------------------------------------------
    async def batch_generate(self, prompts: list[str], concurrency: int = 5) -> list[str]:
        semaphore = asyncio.Semaphore(concurrency)

        async def _guarded(prompt: str) -> str:
            async with semaphore:
                return await self.generate(prompt)

        return await asyncio.gather(*[_guarded(p) for p in prompts])

    # -- Sync wrapper for notebooks and DSPy ----------------------------------
    def generate_sync(self, prompt: str) -> str:
        """Blocking wrapper that works in both CLI and Jupyter (nested loop)."""
        try:
            # Check if an event loop is already running (e.g. Jupyter, DSPy internals)
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # Already inside an event loop — use nest_asyncio to allow nesting
            import nest_asyncio
            nest_asyncio.apply()
            return loop.run_until_complete(self.generate(prompt))
        else:
            return asyncio.run(self.generate(prompt))

    # -- Response extractor --------------------------------------------------
    @staticmethod
    def _extract_message(data: dict) -> str:
        try:
            return (
                data["outputs"][0]
                ["outputs"][0]
                ["outputs"]["message"]["message"]
            )
        except (KeyError, IndexError, TypeError) as e:
            raw = json.dumps(data, indent=2)
            raise ValueError(
                f"Could not extract message from response. Raw:\n{raw}"
            ) from e


# -- Quick smoke-test -------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    async def _test():
        client = LLMControlsClient()
        print("Testing LLMControls API ...")
        t0 = time.time()
        reply = await client.generate("What is the capital of France? Answer in one word.")
        elapsed = time.time() - t0
        print(f"Response ({elapsed:.2f}s): {reply}")

    asyncio.run(_test())
