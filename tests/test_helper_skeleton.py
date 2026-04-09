"""Tests for MarketStructureHelper's empty state, constructor, and instance isolation.

Scope: constructor wiring and the guarantees a freshly constructed helper
makes to its callers. Ingest behavior lives in ``test_register_candle.py``.
"""

import pytest

from market_structure import MarketStructureHelper


class TestEmptyHelper:
    def test_wave_registry_is_empty_tuple(self) -> None:
        h = MarketStructureHelper()
        assert h.wave_registry == ()

    def test_get_last_top_returns_none(self) -> None:
        h = MarketStructureHelper()
        assert h.get_last_top() is None

    def test_get_last_bottom_returns_none(self) -> None:
        h = MarketStructureHelper()
        assert h.get_last_bottom() is None

    def test_get_current_wave_returns_none(self) -> None:
        h = MarketStructureHelper()
        assert h.get_current_wave() is None


class TestConstructor:
    def test_default_histogram_key(self) -> None:
        h = MarketStructureHelper()
        assert h.histogram_key == "tsi_hist"

    def test_default_max_waves(self) -> None:
        h = MarketStructureHelper()
        assert h.max_waves == 200

    def test_custom_histogram_key(self) -> None:
        h = MarketStructureHelper(histogram_key="macd_hist")
        assert h.histogram_key == "macd_hist"

    def test_custom_max_waves(self) -> None:
        h = MarketStructureHelper(max_waves=50)
        assert h.max_waves == 50

    def test_config_is_keyword_only(self) -> None:
        """All config must be passed by keyword, never positionally.

        The ``*`` in the signature makes every following parameter
        keyword-only. This is load-bearing for API stability: adding
        a new config knob later cannot break existing callers by
        reshuffling positional slots.
        """
        with pytest.raises(TypeError):
            MarketStructureHelper("tsi_hist")  # type: ignore[misc]


class TestInstanceIsolation:
    """Lock in that two helpers have fully independent mutable state."""

    def test_mutation_does_not_bleed_across_instances(self) -> None:
        """Modifying one helper's internal list must not affect another.

        Classic Python footgun: declaring ``_wave_registry: list[Wave] = []``
        at *class body* level (instead of inside ``__init__``) creates ONE
        shared list across every instance. This test fails loudly if the
        initialization is ever accidentally hoisted out of ``__init__``.
        """
        h1 = MarketStructureHelper()
        h2 = MarketStructureHelper()

        assert len(h1.wave_registry) == 0
        assert len(h2.wave_registry) == 0

        # Poke at private state — we're testing the underlying list identity,
        # not the public snapshot property.
        h1._wave_registry.append(None)  # type: ignore[arg-type]

        assert len(h1.wave_registry) == 1
        assert len(h2.wave_registry) == 0
