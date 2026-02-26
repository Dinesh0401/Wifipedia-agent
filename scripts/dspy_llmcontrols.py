# =============================================================================
# dspy_llmcontrols.py  -- DSPy LM adapter for LLMControls API
# Bridges DSPy's LM interface with LLMControlsClient so that
# MIPROv2 and ACE faithfulness evaluator can call LLMControls.
# =============================================================================

import logging
from typing import Any

import dspy
from litellm.types.utils import ModelResponse, Choices, Message, Usage

from scripts.llmcontrols_client import LLMControlsClient

logger = logging.getLogger(__name__)


class LLMControlsLM(dspy.LM):
    """
    Custom DSPy LM that routes all calls through LLMControlsClient
    instead of litellm. Returns a litellm-compatible ModelResponse
    so DSPy's _process_completion works unchanged.
    """

    def __init__(self, temperature: float = 0.0, max_tokens: int = 512):
        # Initialize parent with a dummy model name; we override forward()
        # so litellm is never actually called.
        super().__init__(
            model="llmcontrols/custom",
            temperature=temperature,
            max_tokens=max_tokens,
        )
        self.client = LLMControlsClient()

    def forward(
        self,
        prompt: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        **kwargs,
    ) -> ModelResponse:
        """
        Override DSPy's forward to call LLMControls instead of litellm.
        Converts the messages list into a single prompt string, sends it
        to LLMControls, and wraps the response in a litellm ModelResponse.
        """
        # Build a single prompt from messages (DSPy sends system + user messages)
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

        # Call LLMControls synchronously (DSPy runs synchronously internally)
        response_text = self.client.generate_sync(combined_prompt)

        # Wrap in litellm ModelResponse for DSPy compatibility
        return ModelResponse(
            id="llmcontrols-response",
            model="llmcontrols/custom",
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


def configure_dspy_with_llmcontrols(temperature: float = 0.0, max_tokens: int = 512):
    """Convenience function to configure DSPy globally with LLMControls."""
    lm = LLMControlsLM(temperature=temperature, max_tokens=max_tokens)
    dspy.configure(lm=lm)
    logger.info("DSPy configured with LLMControls adapter")
    return lm
