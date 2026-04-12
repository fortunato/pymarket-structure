"""market_structure — swings, trends, support/resistance zones for OHLCV frames."""

from market_structure.freqtrade import attach_market_structure
from market_structure.helper import MarketStructureHelper


def hello() -> str:
    return "Hello from market-structure!"


__all__ = ["MarketStructureHelper", "attach_market_structure", "hello"]
