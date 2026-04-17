"""Tests for the public market_structure dataclasses."""

import dataclasses

import pytest

from market_structure.types import Candle, Pullback, Wave, Zone


def _candle(**overrides: float) -> Candle:
    defaults: dict[str, float] = {
        "open_time": 1_000,
        "open": 100.0,
        "high": 101.0,
        "low": 99.0,
        "close": 100.5,
        "volume": 1.0,
    }
    return Candle(**{**defaults, **overrides})  # type: ignore[arg-type]


class TestCandle:
    def test_is_frozen(self) -> None:
        c = _candle()
        with pytest.raises(dataclasses.FrozenInstanceError):
            c.close = 999.0  # type: ignore[misc]

    def test_has_slots_no_dict(self) -> None:
        c = _candle()
        assert not hasattr(c, "__dict__")

    def test_equality_by_value(self) -> None:
        assert _candle() == _candle()
        assert _candle(close=100.0) != _candle(close=101.0)


class TestWave:
    def _wave(self, **overrides: object) -> Wave:
        c = _candle()
        defaults: dict[str, object] = {
            "id": "w1",
            "side": "up",
            "formation_bar_index": 10,
            "high": c,
            "low": c,
            "highest_close": c,
            "lowest_close": c,
            "highest_close_or_open": c,
            "lowest_close_or_open": c,
            "high_idx": 5,
            "low_idx": 5,
            "highest_close_or_open_idx": 5,
            "lowest_close_or_open_idx": 5,
        }
        return Wave(**{**defaults, **overrides})  # type: ignore[arg-type]

    def test_defaults_have_no_pullback(self) -> None:
        w = self._wave()
        assert w.pullback is None
        assert w.high_since == 0
        assert w.candles == ()

    def test_is_frozen(self) -> None:
        w = self._wave()
        with pytest.raises(dataclasses.FrozenInstanceError):
            w.high_since = 42  # type: ignore[misc]


class TestPullback:
    def test_correction_factor_allows_none(self) -> None:
        p = Pullback(
            length=5,
            breakout_level=100.0,
            price_diff=-2.0,
            correction_factor=None,
            atr_factor=None,
        )
        assert p.correction_factor is None


class TestZone:
    def test_range_is_tuple_not_list(self) -> None:
        z = Zone(
            range=(99.0, 101.0),
            wick_range=(99.0, 101.0),
            anchor_wave_id="w1",
            overlapping_low_wave_ids=(),
            overlapping_high_wave_ids=(),
            is_double=False,
            side="down",
        )
        assert isinstance(z.range, tuple)
