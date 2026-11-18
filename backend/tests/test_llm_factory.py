"""Gateway construction from settings: one route per (model, key), best model first."""

from pydantic import SecretStr

from app.core.config import Settings
from app.llm.factory import build_gateway


async def test_each_model_is_tried_on_every_key_before_the_next_model() -> None:
    s = Settings(
        groq_api_key=SecretStr("k1"),
        groq_api_key_2=SecretStr("k2"),
        groq_api_key_3=SecretStr(""),  # blank keys are ignored
        llm_models=["big", "small"],
    )
    gateway, pool = build_gateway(s)
    try:
        assert [r.name for r in gateway.routes] == [
            "big@key1",
            "big@key2",
            "small@key1",
            "small@key2",
        ]
        assert len(pool.providers) == 2
    finally:
        await pool.aclose()


async def test_single_key_keeps_plain_model_names() -> None:
    s = Settings(groq_api_key=SecretStr("k1"), groq_api_key_2=None, groq_api_key_3=None)
    gateway, pool = build_gateway(s, ["judge"])
    try:
        assert [r.name for r in gateway.routes] == ["judge"]
    finally:
        await pool.aclose()
