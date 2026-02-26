# =============================================================================
# llm_client.py  -- Unified Async LLM Client
# Routes to LLMControls, Claude (Anthropic), or DeepSeek (OpenRouter)
# based on cfg.active_model.
# =============================================================================

import aiohttp
import asyncio
import json
import logging
from typing import Optional

from scripts.config import cfg

logger = logging.getLogger(__name__)


class UnifiedLLMClient:
    """
    Async LLM client that dispatches to the active model backend.

    Backends:
      - llmcontrols: Custom LLMControls API (nested response format)
      - claude:      Anthropic Messages API (https://api.anthropic.com/v1/messages)
      - deepseek:    OpenRouter OpenAI-compatible API (https://openrouter.ai/api/v1)
    """

    def __init__(self, max_retries: int = 3, retry_delay: float = 2.0):
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    # -- Core async generate ---------------------------------------------------
    async def generate(self, prompt: str, timeout: int = 60) -> str:
        backend = cfg.active_model
        if backend == "llmcontrols":
            return await self._generate_llmcontrols(prompt, timeout)
        elif backend == "claude":
            return await self._generate_claude(prompt, timeout)
        elif backend == "deepseek":
            return await self._generate_deepseek(prompt, timeout)
        else:
            raise ValueError(f"Unknown active_model: {backend}")

    # -- Sync wrapper ----------------------------------------------------------
    def generate_sync(self, prompt: str) -> str:
        """Blocking wrapper that works in both CLI and Jupyter (nested loop)."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import nest_asyncio
            nest_asyncio.apply()
            return loop.run_until_complete(self.generate(prompt))
        else:
            return asyncio.run(self.generate(prompt))

    # -- Batch async generate --------------------------------------------------
    async def batch_generate(self, prompts: list[str], concurrency: int = 5) -> list[str]:
        semaphore = asyncio.Semaphore(concurrency)

        async def _guarded(prompt: str) -> str:
            async with semaphore:
                return await self.generate(prompt)

        return await asyncio.gather(*[_guarded(p) for p in prompts])

    # ==========================================================================
    # Backend: LLMControls
    # ==========================================================================
    async def _generate_llmcontrols(self, prompt: str, timeout: int) -> str:
        headers = {
            "Content-Type": "application/json",
            "x-api-key": cfg.llmc_api_key,
        }
        payload = {
            "input_value": prompt,
            "output_type": "chat",
            "input_type": "chat",
        }
        return await self._request_with_retry(
            url=cfg.llmc_api_url,
            headers=headers,
            payload=payload,
            timeout=timeout,
            extractor=self._extract_llmcontrols,
            label="LLMControls",
        )

    @staticmethod
    def _extract_llmcontrols(data: dict) -> str:
        try:
            return (
                data["outputs"][0]
                ["outputs"][0]
                ["outputs"]["message"]["message"]
            )
        except (KeyError, IndexError, TypeError) as e:
            raw = json.dumps(data, indent=2)
            raise ValueError(f"LLMControls: Could not extract message.\n{raw}") from e

    # ==========================================================================
    # Backend: Claude (Anthropic Messages API)
    # ==========================================================================
    async def _generate_claude(self, prompt: str, timeout: int) -> str:
        headers = {
            "Content-Type": "application/json",
            "x-api-key": cfg.anthropic_api_key,
            "anthropic-version": "2023-06-01",
        }
        payload = {
            "model": cfg.claude_model,
            "max_tokens": 512,
            "temperature": cfg.temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        return await self._request_with_retry(
            url="https://api.anthropic.com/v1/messages",
            headers=headers,
            payload=payload,
            timeout=timeout,
            extractor=self._extract_claude,
            label="Claude",
        )

    @staticmethod
    def _extract_claude(data: dict) -> str:
        try:
            return data["content"][0]["text"]
        except (KeyError, IndexError, TypeError) as e:
            raw = json.dumps(data, indent=2)
            raise ValueError(f"Claude: Could not extract message.\n{raw}") from e

    # ==========================================================================
    # Backend: DeepSeek (OpenRouter, OpenAI-compatible)
    # ==========================================================================
    async def _generate_deepseek(self, prompt: str, timeout: int) -> str:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {cfg.openrouter_api_key}",
        }
        payload = {
            "model": cfg.deepseek_model,
            "temperature": cfg.temperature,
            "max_tokens": 512,
            "messages": [{"role": "user", "content": prompt}],
        }
        return await self._request_with_retry(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            payload=payload,
            timeout=timeout,
            extractor=self._extract_deepseek,
            label="DeepSeek",
        )

    @staticmethod
    def _extract_deepseek(data: dict) -> str:
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            raw = json.dumps(data, indent=2)
            raise ValueError(f"DeepSeek: Could not extract message.\n{raw}") from e

    # ==========================================================================
    # Shared retry logic
    # ==========================================================================
    async def _request_with_retry(
        self,
        url: str,
        headers: dict,
        payload: dict,
        timeout: int,
        extractor,
        label: str,
    ) -> str:
        for attempt in range(1, self.max_retries + 1):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        url,
                        json=payload,
                        headers=headers,
                        timeout=aiohttp.ClientTimeout(total=timeout),
                    ) as resp:
                        resp.raise_for_status()
                        data = await resp.json()
                        return extractor(data)
            except aiohttp.ClientResponseError as e:
                logger.warning(f"[{label} attempt {attempt}] HTTP {e.status}: {e.message}")
            except asyncio.TimeoutError:
                logger.warning(f"[{label} attempt {attempt}] Request timed out")
            except Exception as e:
                logger.warning(f"[{label} attempt {attempt}] Error: {e}")

            if attempt < self.max_retries:
                await asyncio.sleep(self.retry_delay * attempt)

        raise RuntimeError(f"{label} API failed after {self.max_retries} attempts")


# -- Quick smoke-test -------------------------------------------------------
if __name__ == "__main__":
    import time
    logging.basicConfig(level=logging.INFO)

    async def _test():
        client = UnifiedLLMClient()
        print(f"Testing with active_model={cfg.active_model} ...")
        t0 = time.time()
        reply = await client.generate("What is the capital of France? Answer in one word.")
        elapsed = time.time() - t0
        print(f"Response ({elapsed:.2f}s): {reply}")

    asyncio.run(_test())
