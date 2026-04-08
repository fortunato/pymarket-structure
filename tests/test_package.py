"""Smoke tests for the market_structure package."""

from market_structure import hello


def test_hello_returns_greeting() -> None:
    assert hello() == "Hello from market-structure!"
