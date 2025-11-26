class LLMServiceError(Exception):
    """General LLM service exception."""
    pass


class LLMRateLimitError(LLMServiceError):
    """Rate limit or quota errors."""
    pass
