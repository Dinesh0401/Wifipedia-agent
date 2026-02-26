# =============================================================================
# dspy_adapter.py  -- DSPy LM adapter for all model backends
# Bridges DSPy's LM interface with UnifiedLLMClient so that
# MIPROv2 and ACE faithfulness evaluator work with any backend.
# =============================================================================

import logging
from typing import Any

import dspy
from litellm.types.utils import ModelResponse, Choices, Message, Usage

from scripts.llm_client import UnifiedLLMClient
from scripts.config import cfg

logger = logging.getLogger(__name__)


class UnifiedDSPyLM(dspy.LM):
    """
    Custom DSPy LM that routes all calls through UnifiedLLMClient.
    Returns a litellm-compatible ModelResponse so DSPy's
    _process_completion works unchanged.
    """

    def __init__(self, temperature: float = 0.0, max_tokens: int = 512):
        model_label = f"{cfg.active_model}/custom"
        super().__init__(
            model=model_label,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        self.client = UnifiedLLMClient()

    def forward(
        self,
        prompt: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        **kwargs,
    ) -> ModelResponse:
        """
        Override DSPy's forward to call the unified client instead of litellm.
        Converts the messages list into a single prompt string, sends it
        to the active backend, and wraps the response in a litellm ModelResponse.
        """
        if messages:
            parts = []
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "system":
                    parts.append(content)
                elif role == "user":
                    parts.append(content)
                elif role == "assistant":
                    parts.append(f"Assistant: {content}")
            combined_prompt = "\n\n".join(parts)
        else:
            combined_prompt = prompt or ""

        response_text = self.client.generate_sync(combined_prompt)

        return ModelResponse(
            id=f"{cfg.active_model}-response",
            model=f"{cfg.active_model}/custom",
            choices=[
                Choices(
                    finish_reason="stop",
                    index=0,
                    message=Message(
                        content=response_text,
                        role="assistant",
                    ),
                )
            ],
            usage=Usage(
                prompt_tokens=len(combined_prompt.split()),
                completion_tokens=len(response_text.split()),
                total_tokens=len(combined_prompt.split()) + len(response_text.split()),
            ),
        )


def configure_dspy(temperature: float = 0.0, max_tokens: int = 512):
    """Convenience function to configure DSPy globally with the active model backend.

    If DSPy is already configured (e.g. from another thread), skip
    reconfiguration to avoid the 'can only be changed by the thread
    that initially configured it' RuntimeError.
    """
    try:
        lm = UnifiedDSPyLM(temperature=temperature, max_tokens=max_tokens)
        dspy.configure(lm=lm)
        logger.info(f"DSPy configured with {cfg.active_model} backend")
        return lm
    except RuntimeError as e:
        if "can only be changed by the thread" in str(e):
            logger.info(f"DSPy already configured, reusing existing settings")
            return dspy.settings.lm
        raise
