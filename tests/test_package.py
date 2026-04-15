"""Smoke tests for the market_structure package."""

from market_structure import MarketStructureHelper, attach_market_structure


def test_public_api_importable() -> None:
    assert MarketStructureHelper is not None
    assert attach_market_structure is not None
