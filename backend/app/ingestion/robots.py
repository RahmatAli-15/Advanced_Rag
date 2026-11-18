"""robots.txt handling per RFC 9309.

Status code rules (RFC 9309 §2.3.1):
  2xx            -> parse and obey the rules
  4xx            -> "unavailable": the crawler may access any resource
  5xx / network  -> "unreachable": assume complete disallow
A 2xx that is not a robots file (e.g. a bot-challenge HTML page) is treated as
unreachable. Being conservative is the polite default.
"""

from dataclasses import dataclass
from enum import StrEnum
from urllib.robotparser import RobotFileParser


class RobotsState(StrEnum):
    PARSED = "parsed"
    UNAVAILABLE = "unavailable"  # allow all
    UNREACHABLE = "unreachable"  # disallow all


@dataclass(frozen=True)
class RobotsPolicy:
    state: RobotsState
    _parser: RobotFileParser | None = None

    def can_fetch(self, user_agent: str, url: str) -> bool:
        if self.state is RobotsState.UNAVAILABLE:
            return True
        if self.state is RobotsState.UNREACHABLE or self._parser is None:
            return False
        return self._parser.can_fetch(user_agent, url)

    def crawl_delay(self, user_agent: str) -> float | None:
        if self._parser is None:
            return None
        delay = self._parser.crawl_delay(user_agent)
        return float(delay) if delay is not None else None


def _looks_like_robots(body: str) -> bool:
    head = body.lstrip()[:200].lower()
    return not head.startswith(("<!doctype", "<html"))


def policy_from_response(status: int | None, body: str) -> RobotsPolicy:
    """Build a policy from a robots.txt fetch. `status=None` means a network error."""
    if status is None or status >= 500:
        return RobotsPolicy(RobotsState.UNREACHABLE)
    if 400 <= status < 500:
        return RobotsPolicy(RobotsState.UNAVAILABLE)
    if not (200 <= status < 300) or not _looks_like_robots(body):
        return RobotsPolicy(RobotsState.UNREACHABLE)
    parser = RobotFileParser()
    parser.parse(body.splitlines())
    return RobotsPolicy(RobotsState.PARSED, parser)
