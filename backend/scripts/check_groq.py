"""Smoke test: is GROQ_API_KEY valid? Lists available models. Never prints the key."""

import sys

import httpx

from app.core.config import get_settings


def main() -> int:
    s = get_settings()
    if s.groq_api_key is None:
        print("GROQ_API_KEY is not set in .env")
        return 1
    r = httpx.get(
        f"{s.groq_base_url}/models",
        headers={"Authorization": f"Bearer {s.groq_api_key.get_secret_value()}"},
        timeout=15,
    )
    if r.status_code != 200:
        print(f"Groq returned HTTP {r.status_code}")
        return 1
    ids = sorted(m["id"] for m in r.json()["data"])
    print(f"Key OK. {len(ids)} models available:")
    for i in ids:
        print(" -", i)
    return 0


if __name__ == "__main__":
    sys.exit(main())
