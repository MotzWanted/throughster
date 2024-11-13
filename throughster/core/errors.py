import httpx


class BaseApiError(Exception):
    """Base API exception."""

    def __init__(self, message: str, *, request: httpx.Request, response: httpx.Response) -> None:
        super().__init__(message)
        self.request = request
        self.response = response


class RateLimitError(BaseApiError):
    """Exception raised when the API rate limit is exceeded."""


class AzureContentFilterError(BaseApiError):
    """Exception raised when content is filtered due to Azure or OpenAI's content management policy."""


class CompletionError(Exception):
    """Exception raised when the completion response fails validation."""

    def __init__(
        self,
        message: str = "Error in generating structured response.",
    ) -> None:
        super().__init__(message)


class StructuredResponseError(Exception):
    """Exception raised when the structured response is invalid."""

    def __init__(
        self,
        message: str = "Error in generating structured response.",
    ) -> None:
        super().__init__(message)
