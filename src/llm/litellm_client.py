import os
import logging
from typing import List, Dict, Union
from litellm import completion, embedding
from src.llm.exceptions import LLMServiceError, LLMRateLimitError
from src.llm.prompt_loader import prompt_loader

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class LiteLLMClient:
    """
    LiteLLM-powered client for dataset analytics & forecasting.
    Handles:
    - Chat completions
    - Prompts from YAML templates
    - Optional embeddings
    - Error normalization
    """

    def __init__(self):
        self.chat_model = os.getenv("LLM_MODEL", "gemini/gemini-2.5-flash-lite")
        self.embed_model = os.getenv("EMBED_MODEL", "gemini/text-embedding-004")

        logger.info("===============================================")
        logger.info(" LiteLLMClient Initialized ")
        logger.info(f" Chat Model      = {self.chat_model}")
        logger.info(f" Embedding Model = {self.embed_model}")
        logger.info("===============================================")

    # ----------------------------------------------------------------------
    # Chat Completion
    # ----------------------------------------------------------------------
    def chat(self, prompt: str, override_model: str = None) -> str:
        try:
            model = override_model or self.chat_model

            response = completion(
                model=model,
                messages=[{"role": "user", "content": prompt}],
            )

            content = (
                response.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )

            if not content:
                raise LLMServiceError("Empty LLM response")

            return content.strip()

        except Exception as e:
            if "rate" in str(e).lower():
                raise LLMRateLimitError(str(e))
            raise LLMServiceError(str(e))

    # ----------------------------------------------------------------------
    # Template-Based Chat
    # ----------------------------------------------------------------------
    def chat_from_template(self, template_key: str, variables: Dict) -> str:
        template = prompt_loader.get(template_key)
        if not template:
            raise LLMServiceError(f"Template '{template_key}' not found")

        for key, val in variables.items():
            template = template.replace(f"{{{{{key}}}}}", str(val))

        return self.chat(template)

    # ----------------------------------------------------------------------
    # Embeddings (optional)
    # ----------------------------------------------------------------------
    def embed(self, text: str) -> List[float]:
        try:
            safe_text = text[:16000]
            resp = embedding(
                model=self.embed_model,
                input=safe_text
            )
            return resp["data"][0]["embedding"]

        except Exception as e:
            raise LLMServiceError(f"Embedding failed: {e}")

    # ----------------------------------------------------------------------
    # Utility
    # ----------------------------------------------------------------------
    def get_embedding_dim(self) -> int:
        try:
            emb = self.embed("dimension probe")
            return len(emb)
        except Exception:
            return 0


# Singleton export
lite_client = LiteLLMClient()
