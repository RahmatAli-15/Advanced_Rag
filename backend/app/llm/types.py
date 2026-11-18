"""Provider-neutral LLM request/response types and errors."""

from dataclasses import dataclass, field
from typing import Literal

Role = Literal["system", "user", "assistant"]


@dataclass(frozen=True)
class Message:
    role: Role
    content: str


@dataclass(frozen=True)
class LLMRequest:
    messages: list[Message]
    temperature: float = 0.0
    max_tokens: int = 1024
    json_mode: bool = False


@dataclass(frozen=True)
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass(frozen=True)
class LLMResponse:
    text: str
    model: str
    provider: str
    usage: Usage = field(default_factory=Usage)
    latency_ms: float = 0.0
    attempts: int = 1  # total provider calls made, including retries and fallbacks


@dataclass(frozen=True)
class StreamEvent:
    """A piece of a streamed completion. The last event has done=True and carries the
    model that answered and the token usage."""

    text: str = ""
    done: bool = False
    model: str = ""
    usage: Usage | None = None


class LLMError(Exception):
    """Base class for provider failures."""


class RateLimited(LLMError):
    def __init__(self, message: str, retry_after_s: float | None = None) -> None:
        super().__init__(message)
        self.retry_after_s = retry_after_s


class TransientError(LLMError):
    """Timeouts, 5xx: worth retrying."""


class FatalError(LLMError):
    """Bad request, auth, unknown model: retrying will not help."""


class AllModelsFailed(LLMError):
    """Every model in the fallback chain failed or was unavailable."""


class StreamInterrupted(LLMError):
    """A stream failed after text had already been sent to the caller. It is not
    retried: splicing a second model's output onto the first would be incoherent."""
