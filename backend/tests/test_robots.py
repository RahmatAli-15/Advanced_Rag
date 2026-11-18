import pytest

from app.ingestion.robots import RobotsState, policy_from_response

UA = "AdvancedRagBot/0.1"


@pytest.mark.parametrize(
    ("status", "state", "allowed"),
    [
        (404, RobotsState.UNAVAILABLE, True),  # RFC 9309: 4xx -> may crawl
        (418, RobotsState.UNAVAILABLE, True),
        (503, RobotsState.UNREACHABLE, False),  # 5xx -> assume full disallow
        (None, RobotsState.UNREACHABLE, False),  # network error
    ],
)
def test_status_codes(status: int | None, state: RobotsState, allowed: bool) -> None:
    policy = policy_from_response(status, "")
    assert policy.state is state
    assert policy.can_fetch(UA, "https://example.org/page") is allowed


def test_rules_are_obeyed() -> None:
    policy = policy_from_response(200, "User-agent: *\nDisallow: /private\nCrawl-delay: 7\n")
    assert policy.state is RobotsState.PARSED
    assert policy.can_fetch(UA, "https://example.org/public")
    assert not policy.can_fetch(UA, "https://example.org/private/x")
    assert policy.crawl_delay(UA) == 7.0


def test_html_challenge_served_as_robots_is_treated_as_unreachable() -> None:
    policy = policy_from_response(200, "<!DOCTYPE html><html>are you human?</html>")
    assert policy.state is RobotsState.UNREACHABLE
    assert not policy.can_fetch(UA, "https://example.org/anything")
