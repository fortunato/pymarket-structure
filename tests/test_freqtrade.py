"""Stage 12 tests for the Freqtrade wrapper (attach_market_structure).

The keystone test is ``TestPerBarParity``: a brute-force ``register_candle``
loop captures helper state at every bar, then ``attach_market_structure``
projects columns over the same data.  The two must match exactly.
"""

# pyright: reportPrivateUsage=false

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from market_structure import MarketStructureHelper
from market_structure.atr import _compute_atr
from market_structure.freqtrade import (
    VALID_COLUMNS,
    MarketStructureDesyncError,
    _score_zone_quality,
    attach_market_structure,
)
from market_structure.types import Candle

# ---------------------------------------------------------------------------
# Fixture loading (same as test_parity.py / test_zones.py)
# ---------------------------------------------------------------------------

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "ms-LTCUSDT-4h.json"
HISTOGRAM_KEY = "tsi_histogram"


def _load_raw() -> list[dict[str, object]]:
    with FIXTURE_PATH.open() as f:
        return json.load(f)


def _to_dataframe(rows: list[dict[str, object]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["open_time"] = (
        pd.to_datetime(df["openTime"]).dt.tz_localize(None).astype("datetime64[ms]").astype("int64")
    )
    cols = ["open_time", "open", "high", "low", "close", "volume", HISTOGRAM_KEY]
    return pd.DataFrame(df[cols])


def _make_candle(row: dict[str, object]) -> Candle:
    return Candle(
        open_time=int(pd.Timestamp(str(row["openTime"]), tz="UTC").value // 10**6),
        open=float(row["open"]),  # type: ignore[arg-type]
        high=float(row["high"]),  # type: ignore[arg-type]
        low=float(row["low"]),  # type: ignore[arg-type]
        close=float(row["close"]),  # type: ignore[arg-type]
        volume=float(row["volume"]),  # type: ignore[arg-type]
        histogram_value=float(row[HISTOGRAM_KEY]),  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------------
# Synthetic helpers
# ---------------------------------------------------------------------------


def _synthetic_df() -> pd.DataFrame:
    """7-bar synthetic frame with 3 confirmed waves + 1 forming."""
    return pd.DataFrame(
        {
            "open_time": [1000, 2000, 3000, 4000, 5000, 6000, 7000],
            "open": [100.0, 102.0, 107.0, 104.0, 99.0, 100.0, 106.0],
            "high": [105.0, 110.0, 109.0, 106.0, 103.0, 108.0, 107.0],
            "low": [98.0, 100.0, 101.0, 97.0, 96.0, 99.0, 95.0],
            "close": [103.0, 108.0, 106.0, 98.0, 101.0, 107.0, 96.0],
            "volume": [1.0] * 7,
            "tsi_hist": [0.4, 0.2, -0.1, -0.3, 0.2, 0.5, -0.4],
        }
    )


# ---------------------------------------------------------------------------
# Per-bar parity (the keystone test)
# ---------------------------------------------------------------------------


def _build_reference(raw: list[dict[str, object]]) -> pd.DataFrame:
    """Brute-force register_candle loop, capturing state at every bar."""
    h = MarketStructureHelper(histogram_key=HISTOGRAM_KEY)
    records: list[dict[str, object]] = []

    # Pre-compute ATR for reference.
    highs_arr = np.array([float(r["high"]) for r in raw])  # type: ignore[arg-type]
    lows_arr = np.array([float(r["low"]) for r in raw])  # type: ignore[arg-type]
    closes_arr = np.array([float(r["close"]) for r in raw])  # type: ignore[arg-type]
    atr_arr = _compute_atr(highs_arr, lows_arr, closes_arr, period=14)

    # Edge-detection state for event columns.
    prev_sfp_h = False
    prev_sfp_l = False
    prev_sbc = False
    prev_ltp = np.nan
    prev_lbp = np.nan
    prev_sbl_val = np.nan
    prev_three_up = False
    prev_three_down = False

    for bar_idx, row in enumerate(raw):
        h.register_candle(_make_candle(row))

        current = h.get_current_wave()
        last_top = h.get_last_top()
        last_bottom = h.get_last_bottom()
        prev_top = h.get_previous_top()
        prev_bottom = h.get_previous_bottom()
        last_confirmed = h._wave_registry[-1] if h._wave_registry else None

        records.append(
            {
                "wave_side": current.side if current else "",
                "is_trending_up": h.is_trending_up(),
                "is_trending_down": h.is_trending_down(),
                "last_top_price": (
                    max(last_top.highest_close_or_open.close, last_top.highest_close_or_open.open)
                    if last_top
                    else np.nan
                ),
                "last_bottom_price": (
                    min(
                        last_bottom.lowest_close_or_open.close,
                        last_bottom.lowest_close_or_open.open,
                    )
                    if last_bottom
                    else np.nan
                ),
                "made_higher_high": (
                    MarketStructureHelper.made_higher_high(last_top, prev_top)
                    if last_top and prev_top
                    else pd.NA
                ),
                "made_lower_high": (
                    MarketStructureHelper.made_lower_high(last_top, prev_top)
                    if last_top and prev_top
                    else pd.NA
                ),
                "made_higher_low": (
                    MarketStructureHelper.made_higher_low(last_bottom, prev_bottom)
                    if last_bottom and prev_bottom
                    else pd.NA
                ),
                "made_lower_low": (
                    MarketStructureHelper.made_lower_low(last_bottom, prev_bottom)
                    if last_bottom and prev_bottom
                    else pd.NA
                ),
                "high_since": last_top.high_since if last_top else pd.NA,
                "low_since": last_bottom.low_since if last_bottom else pd.NA,
                "pullback_length": (
                    last_confirmed.pullback.length
                    if last_confirmed and last_confirmed.pullback
                    else pd.NA
                ),
                "pullback_correction_factor": (
                    last_confirmed.pullback.correction_factor
                    if last_confirmed
                    and last_confirmed.pullback
                    and last_confirmed.pullback.correction_factor is not None
                    else np.nan
                ),
                "wave_length": (len(last_confirmed.candles) if last_confirmed else pd.NA),
                "pullback_breakout_level": (
                    last_confirmed.pullback.breakout_level
                    if last_confirmed and last_confirmed.pullback
                    else np.nan
                ),
                "pullback_price_diff": (
                    last_confirmed.pullback.price_diff
                    if last_confirmed and last_confirmed.pullback
                    else np.nan
                ),
                "bearish_divergence": (
                    MarketStructureHelper.is_diverging(last_top, prev_top)
                    if last_top and prev_top
                    else pd.NA
                ),
                "bullish_divergence": (
                    MarketStructureHelper.is_diverging(last_bottom, prev_bottom)
                    if last_bottom and prev_bottom
                    else pd.NA
                ),
                "wave_count": len(h._wave_registry),
                "forming_wave_high": current.high.high if current else np.nan,
                "forming_wave_low": current.low.low if current else np.nan,
            }
        )

        # Zone columns — query helper after each bar (brute-force reference).
        # Pass ``atr_arr`` to match the projected path (``_snapshot_zones``
        # in freqtrade.py), which now drives the tolerance predicate via
        # the ATR multiple rather than percentage-of-price fallback.
        sup_zones = h.get_support_zones(atr_arr=atr_arr)
        sz = sup_zones[0] if sup_zones else None
        res_zones = h.get_resistance_zones(atr_arr=atr_arr)
        rz = res_zones[0] if res_zones else None
        records[-1].update(
            {
                "support_zone_low": sz.range[0] if sz else np.nan,
                "support_zone_high": sz.range[1] if sz else np.nan,
                "support_zone_wick_low": sz.wick_range[0] if sz else np.nan,
                "support_zone_wick_high": sz.wick_range[1] if sz else np.nan,
                "support_is_double": sz.is_double if sz else pd.NA,
                "support_overlap_count": len(sz.overlapping_low_wave_ids) if sz else pd.NA,
                "resistance_zone_low": rz.range[0] if rz else np.nan,
                "resistance_zone_high": rz.range[1] if rz else np.nan,
                "resistance_zone_wick_low": rz.wick_range[0] if rz else np.nan,
                "resistance_zone_wick_high": rz.wick_range[1] if rz else np.nan,
                "resistance_is_double": rz.is_double if rz else pd.NA,
                "resistance_overlap_count": (len(rz.overlapping_high_wave_ids) if rz else pd.NA),
            }
        )

        # Zone quality — computed from zone objects with bars_since_formed
        # relative to the zone's anchor wave formation bar.
        zq_s = np.nan
        zq_r = np.nan
        if last_confirmed:
            fb = last_confirmed.formation_bar_index
            atr_at_fb = atr_arr[fb] if fb < len(atr_arr) else np.nan
            wave_fb_by_id = {w.id: w.formation_bar_index for w in h._wave_registry}
            if sz:
                anchor_fb = wave_fb_by_id.get(sz.anchor_wave_id, fb)
                zq_s = _score_zone_quality(sz, atr_at_fb, fb - anchor_fb)
            if rz:
                anchor_fb = wave_fb_by_id.get(rz.anchor_wave_id, fb)
                zq_r = _score_zone_quality(rz, atr_at_fb, fb - anchor_fb)

        # Pullback ATR factor — from the last confirmed wave's pullback.
        pb_atr = np.nan
        if last_confirmed and last_confirmed.pullback:
            fb = last_confirmed.formation_bar_index
            atr_at_fb = atr_arr[fb] if fb < len(atr_arr) else np.nan
            if not np.isnan(atr_at_fb) and atr_at_fb > 0:
                pb_atr = abs(last_confirmed.pullback.price_diff) / atr_at_fb

        # Wave volume and amplitude ratios — from last confirmed wave.
        wave_vol = np.nan
        wave_vol_ratio = np.nan
        wave_amp_ratio = np.nan
        if last_confirmed:
            wave_vol = sum(c.volume for c in last_confirmed.candles)
            # Previous same-direction wave for ratios.
            same_dir = h._top_waves if last_confirmed.side == "up" else h._bottom_waves
            prev_same = same_dir[-2] if len(same_dir) >= 2 else None
            if prev_same is not None:
                prev_vol = sum(c.volume for c in prev_same.candles)
                wave_vol_ratio = wave_vol / prev_vol if prev_vol > 0 else np.nan
                # Amplitude ratio (ATR-normalized).
                fb = last_confirmed.formation_bar_index
                atr_cur = atr_arr[fb] if fb < len(atr_arr) else np.nan
                cur_amp = (
                    (last_confirmed.high.high - last_confirmed.low.low) / atr_cur
                    if not np.isnan(atr_cur) and atr_cur > 0
                    else np.nan
                )
                pfb = prev_same.formation_bar_index
                atr_prev = atr_arr[pfb] if pfb < len(atr_arr) else np.nan
                prev_amp = (
                    (prev_same.high.high - prev_same.low.low) / atr_prev
                    if not np.isnan(atr_prev) and atr_prev > 0
                    else np.nan
                )
                wave_amp_ratio = (
                    cur_amp / prev_amp
                    if not np.isnan(cur_amp) and not np.isnan(prev_amp) and prev_amp > 0
                    else np.nan
                )

        # Trend wave count — consecutive HH+HL or LH+LL pairs backward.
        twc = 0
        t_waves = h._top_waves
        b_waves = h._bottom_waves
        if len(t_waves) >= 2 and len(b_waves) >= 2:
            ti = len(t_waves) - 1
            bi = len(b_waves) - 1
            while ti >= 1 and bi >= 1:
                t_hh = max(
                    t_waves[ti].highest_close_or_open.close, t_waves[ti].highest_close_or_open.open
                ) > max(
                    t_waves[ti - 1].highest_close_or_open.close,
                    t_waves[ti - 1].highest_close_or_open.open,
                )
                b_hl = min(
                    b_waves[bi].lowest_close_or_open.close, b_waves[bi].lowest_close_or_open.open
                ) > min(
                    b_waves[bi - 1].lowest_close_or_open.close,
                    b_waves[bi - 1].lowest_close_or_open.open,
                )
                t_lh = max(
                    t_waves[ti].highest_close_or_open.close, t_waves[ti].highest_close_or_open.open
                ) < max(
                    t_waves[ti - 1].highest_close_or_open.close,
                    t_waves[ti - 1].highest_close_or_open.open,
                )
                b_ll = min(
                    b_waves[bi].lowest_close_or_open.close, b_waves[bi].lowest_close_or_open.open
                ) < min(
                    b_waves[bi - 1].lowest_close_or_open.close,
                    b_waves[bi - 1].lowest_close_or_open.open,
                )
                if (t_hh and b_hl) or (t_lh and b_ll):
                    twc += 1
                else:
                    break
                ti -= 1
                bi -= 1

        # Three-push patterns.
        three_up = False
        if len(t_waves) >= 3:
            tw1, tw2, tw3 = t_waves[-3], t_waves[-2], t_waves[-1]
            hv1 = max(tw1.highest_close_or_open.close, tw1.highest_close_or_open.open)
            hv2 = max(tw2.highest_close_or_open.close, tw2.highest_close_or_open.open)
            hv3 = max(tw3.highest_close_or_open.close, tw3.highest_close_or_open.open)
            ta1 = tw1.high.high - tw1.low.low
            ta2 = tw2.high.high - tw2.low.low
            ta3 = tw3.high.high - tw3.low.low
            if ta3 < ta2 < ta1 and hv3 > hv2 > hv1 and (hv3 - hv2) < (hv2 - hv1):
                three_up = True

        three_down = False
        if len(b_waves) >= 3:
            bw1, bw2, bw3 = b_waves[-3], b_waves[-2], b_waves[-1]
            lv1 = min(bw1.lowest_close_or_open.close, bw1.lowest_close_or_open.open)
            lv2 = min(bw2.lowest_close_or_open.close, bw2.lowest_close_or_open.open)
            lv3 = min(bw3.lowest_close_or_open.close, bw3.lowest_close_or_open.open)
            ba1 = bw1.high.high - bw1.low.low
            ba2 = bw2.high.high - bw2.low.low
            ba3 = bw3.high.high - bw3.low.low
            if ba3 < ba2 < ba1 and lv3 < lv2 < lv1 and (lv2 - lv3) < (lv1 - lv2):
                three_down = True

        # three_push — edge-detected: fire only on the first bar of each True span.
        three_up_edge = three_up and not prev_three_up
        three_down_edge = three_down and not prev_three_down
        prev_three_up = three_up
        prev_three_down = three_down

        records[-1].update(
            {
                "zone_quality_support": zq_s,
                "zone_quality_resistance": zq_r,
                "pullback_atr_factor": pb_atr,
                "wave_volume": wave_vol,
                "wave_volume_ratio": wave_vol_ratio,
                "wave_amplitude_ratio": wave_amp_ratio,
                "trend_wave_count": twc,
                "three_push_up": three_up_edge,
                "three_push_down": three_down_edge,
            }
        )

        # ATR, structure break, distance, SFP columns.
        atr_val = atr_arr[bar_idx]

        # structure_break_level: uptrend (HH+HL) → last_bottom LCO, downtrend (LH+LL) → last_top HCO
        sbl = np.nan
        if last_top and last_bottom and prev_top and prev_bottom:
            lt_hco = max(last_top.highest_close_or_open.close, last_top.highest_close_or_open.open)
            pt_hco = max(prev_top.highest_close_or_open.close, prev_top.highest_close_or_open.open)
            lb_lco = min(
                last_bottom.lowest_close_or_open.close, last_bottom.lowest_close_or_open.open
            )
            pb_lco = min(
                prev_bottom.lowest_close_or_open.close, prev_bottom.lowest_close_or_open.open
            )
            hh = lt_hco > pt_hco
            hl = lb_lco > pb_lco
            lh = lt_hco < pt_hco
            ll = lb_lco < pb_lco
            if hh and hl:
                sbl = lb_lco
            elif lh and ll:
                sbl = lt_hco

        # wave_amplitude and wave_slope: from last confirmed wave
        wave_amp = np.nan
        wave_slp = np.nan
        if last_confirmed:
            fb = last_confirmed.formation_bar_index
            atr_at_fb = atr_arr[fb] if fb < len(atr_arr) else np.nan
            if not np.isnan(atr_at_fb) and atr_at_fb > 0:
                wave_amp = (last_confirmed.high.high - last_confirmed.low.low) / atr_at_fb
                wlen = len(last_confirmed.candles)
                if wlen > 0:
                    wave_slp = wave_amp / wlen

        # distance_to_support/resistance
        close_val = float(row["close"])  # type: ignore[arg-type]
        dist_s = np.nan
        dist_r = np.nan
        if sz and not np.isnan(atr_val) and atr_val > 0:
            dist_s = (close_val - sz.range[1]) / atr_val
        if rz and not np.isnan(atr_val) and atr_val > 0:
            dist_r = (rz.range[0] - close_val) / atr_val

        # sfp_high/low — edge-detected: fire only on first bar of each cluster,
        # or when the reference level changes.
        high_val = float(row["high"])  # type: ignore[arg-type]
        low_val = float(row["low"])  # type: ignore[arg-type]
        sfp_h_raw = False
        sfp_l_raw = False
        cur_ltp = np.nan
        cur_lbp = np.nan
        if last_top:
            cur_ltp = max(last_top.highest_close_or_open.close, last_top.highest_close_or_open.open)
            sfp_h_raw = high_val > cur_ltp and close_val < cur_ltp
        if last_bottom:
            cur_lbp = min(
                last_bottom.lowest_close_or_open.close, last_bottom.lowest_close_or_open.open
            )
            sfp_l_raw = low_val < cur_lbp and close_val > cur_lbp

        ltp_changed = cur_ltp != prev_ltp and not (np.isnan(cur_ltp) and np.isnan(prev_ltp))
        lbp_changed = cur_lbp != prev_lbp and not (np.isnan(cur_lbp) and np.isnan(prev_lbp))
        sfp_h = sfp_h_raw and (not prev_sfp_h or ltp_changed)
        sfp_l = sfp_l_raw and (not prev_sfp_l or lbp_changed)
        prev_sfp_h = sfp_h_raw
        prev_sfp_l = sfp_l_raw
        prev_ltp = cur_ltp
        prev_lbp = cur_lbp

        # structure_break_confirmed — edge-detected
        sbc_raw = False
        if not np.isnan(sbl):
            is_up = last_top and last_bottom and prev_top and prev_bottom and hh and hl  # type: ignore[possibly-undefined]
            is_down = last_top and last_bottom and prev_top and prev_bottom and lh and ll  # type: ignore[possibly-undefined]
            if (is_up and close_val < sbl) or (is_down and close_val > sbl):
                sbc_raw = True

        sbl_changed = sbl != prev_sbl_val and not (np.isnan(sbl) and np.isnan(prev_sbl_val))
        sbc = sbc_raw and (not prev_sbc or sbl_changed)
        prev_sbc = sbc_raw
        prev_sbl_val = sbl

        records[-1].update(
            {
                "atr": atr_val,
                "structure_break_level": sbl,
                "wave_amplitude": wave_amp,
                "wave_slope": wave_slp,
                "distance_to_support": dist_s,
                "distance_to_resistance": dist_r,
                "sfp_high": sfp_h,
                "sfp_low": sfp_l,
                "structure_break_confirmed": sbc,
            }
        )

    # bars_since_last_top / bars_since_last_bottom — need second pass
    # Re-walk wave registry to find formation bars of tops/bottoms
    h2 = MarketStructureHelper(histogram_key=HISTOGRAM_KEY)
    top_confirm_bars: set[int] = set()
    bottom_confirm_bars: set[int] = set()
    prev_wave_count = 0
    for _bar_idx, row in enumerate(raw):
        h2.register_candle(_make_candle(row))
        cur_count = len(h2._wave_registry)
        if cur_count > prev_wave_count:
            new_wave = h2._wave_registry[-1]
            if new_wave.side == "up":
                top_confirm_bars.add(new_wave.formation_bar_index)
            else:
                bottom_confirm_bars.add(new_wave.formation_bar_index)
            prev_wave_count = cur_count

    counter_top: int | None = None
    counter_bottom: int | None = None
    for i in range(len(raw)):
        if i in top_confirm_bars:
            counter_top = 0
        records[i]["bars_since_last_top"] = counter_top if counter_top is not None else pd.NA
        if counter_top is not None:
            counter_top += 1

        if i in bottom_confirm_bars:
            counter_bottom = 0
        records[i]["bars_since_last_bottom"] = (
            counter_bottom if counter_bottom is not None else pd.NA
        )
        if counter_bottom is not None:
            counter_bottom += 1

    # Zone lifecycle — forward pass state machine (mirrors _project_zone_lifecycle).
    sup_state = "intact"
    res_state = "intact"
    sup_zone: tuple[float, float] | None = None
    res_zone: tuple[float, float] | None = None
    sup_retest_count = 0
    res_retest_count = 0

    for i in range(len(raw)):
        r = records[i]
        c = float(raw[i]["close"])  # type: ignore[arg-type]
        h_val = float(raw[i]["high"])  # type: ignore[arg-type]

        s_low = r["support_zone_low"]
        cur_sup: tuple[float, float] | None = (
            (float(s_low), float(r["support_zone_high"]))  # type: ignore[arg-type]
            if not (isinstance(s_low, float) and np.isnan(s_low))
            else None
        )
        r_low = r["resistance_zone_low"]
        cur_res: tuple[float, float] | None = (
            (float(r_low), float(r["resistance_zone_high"]))  # type: ignore[arg-type]
            if not (isinstance(r_low, float) and np.isnan(r_low))
            else None
        )

        if cur_sup != sup_zone:
            sup_zone = cur_sup
            sup_state = "intact"
            sup_retest_count = 0
        if cur_res != res_zone:
            res_zone = cur_res
            res_state = "intact"
            res_retest_count = 0

        brk_s = False
        brk_r = False
        ret_s = False
        ret_r = False
        flp_s = False
        flp_r = False
        fr_s = False
        fr_r = False

        if sup_zone is not None:
            z_low, z_high = sup_zone
            if sup_state == "intact" and c < z_low:
                sup_state = "broken"
                brk_s = True
            elif sup_state == "broken" and h_val >= z_low:
                sup_state = "retested"
                sup_retest_count += 1
                ret_s = True
            elif sup_state == "retested":
                if c < z_low:
                    sup_state = "intact"
                    flp_s = True
                    sup_retest_count = 0
                elif c > z_high:
                    sup_state = "intact"
                    fr_s = True
                    sup_retest_count = 0

        if res_zone is not None:
            z_low, z_high = res_zone
            if res_state == "intact" and c > z_high:
                res_state = "broken"
                brk_r = True
            elif res_state == "broken" and float(raw[i]["low"]) <= z_high:  # type: ignore[arg-type]
                res_state = "retested"
                res_retest_count += 1
                ret_r = True
            elif res_state == "retested":
                if c > z_high:
                    res_state = "intact"
                    flp_r = True
                    res_retest_count = 0
                elif c < z_low:
                    res_state = "intact"
                    fr_r = True
                    res_retest_count = 0

        r["zone_break_support"] = brk_s
        r["zone_break_resistance"] = brk_r
        r["zone_retest_support"] = ret_s
        r["zone_retest_resistance"] = ret_r
        r["zone_retest_count_support"] = sup_retest_count
        r["zone_retest_count_resistance"] = res_retest_count
        r["zone_flip_support"] = flp_s
        r["zone_flip_resistance"] = flp_r
        r["zone_failed_retest_support"] = fr_s
        r["zone_failed_retest_resistance"] = fr_r

    # Trend duration — per-bar counter since trend epoch started.
    # Trend epoch: the bar where is_trending_up or is_trending_down first became True.
    ref_epoch: int | None = None
    prev_was_trending = False
    for i in range(len(raw)):
        is_trending = records[i]["is_trending_up"] or records[i]["is_trending_down"]
        if is_trending and not prev_was_trending:
            ref_epoch = i
        elif not is_trending:
            ref_epoch = None
        records[i]["trend_duration"] = (
            i - ref_epoch if ref_epoch is not None and is_trending else pd.NA
        )
        prev_was_trending = is_trending

    return pd.DataFrame(records)


@pytest.fixture()
def parity_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build both brute-force reference and projected DataFrames."""
    raw = _load_raw()
    df = _to_dataframe(raw)

    reference = _build_reference(raw)

    store: dict[str, MarketStructureHelper] = {}
    projected, _ = attach_market_structure(
        df,
        {"pair": "LTC/USDT"},
        store,
        hist_col=HISTOGRAM_KEY,
    )
    return reference, projected


class TestPerBarParity:
    """Brute-force per-bar reference vs projected columns.

    Each test validates one column for clear failure diagnostics.
    """

    def test_wave_side(self, parity_data: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        ref, proj = parity_data
        assert list(ref["wave_side"]) == list(proj["ms_wave_side"])

    def test_is_trending_up(self, parity_data: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        ref, proj = parity_data
        assert list(ref["is_trending_up"]) == list(proj["ms_is_trending_up"])

    def test_is_trending_down(self, parity_data: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        ref, proj = parity_data
        assert list(ref["is_trending_down"]) == list(proj["ms_is_trending_down"])

    def test_last_top_price(self, parity_data: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        ref, proj = parity_data
        ref_vals = ref["last_top_price"].to_numpy(dtype=float)
        proj_vals = proj["ms_last_top_price"].to_numpy(dtype=float)
        np.testing.assert_array_equal(ref_vals, proj_vals)

    def test_last_bottom_price(self, parity_data: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        ref, proj = parity_data
        ref_vals = ref["last_bottom_price"].to_numpy(dtype=float)
        proj_vals = proj["ms_last_bottom_price"].to_numpy(dtype=float)
        np.testing.assert_array_equal(ref_vals, proj_vals)

    def test_made_higher_high(self, parity_data: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        ref, proj = parity_data
        ref_s = pd.array(ref["made_higher_high"], dtype="boolean")  # type: ignore[arg-type]
        proj_s = proj["ms_made_higher_high"]
        pd.testing.assert_extension_array_equal(ref_s, proj_s.array)  # type: ignore[arg-type]

    def test_made_lower_high(self, parity_data: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        ref, proj = parity_data
        ref_s = pd.array(ref["made_lower_high"], dtype="boolean")  # type: ignore[arg-type]
        proj_s = proj["ms_made_lower_high"]
        pd.testing.assert_extension_array_equal(ref_s, proj_s.array)  # type: ignore[arg-type]

    def test_made_higher_low(self, parity_data: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        ref, proj = parity_data
        ref_s = pd.array(ref["made_higher_low"], dtype="boolean")  # type: ignore[arg-type]
        proj_s = proj["ms_made_higher_low"]
        pd.testing.assert_extension_array_equal(ref_s, proj_s.array)  # type: ignore[arg-type]

    def test_made_lower_low(self, parity_data: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        ref, proj = parity_data
        ref_s = pd.array(ref["made_lower_low"], dtype="boolean")  # type: ignore[arg-type]
        proj_s = proj["ms_made_lower_low"]
        pd.testing.assert_extension_array_equal(ref_s, proj_s.array)  # type: ignore[arg-type]

    def test_high_since(self, parity_data: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        ref, proj = parity_data
        ref_s = pd.array(ref["high_since"], dtype="Int32")  # type: ignore[arg-type]
        proj_s = proj["ms_high_since"]
        pd.testing.assert_extension_array_equal(ref_s, proj_s.array)  # type: ignore[arg-type]

    def test_low_since(self, parity_data: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        ref, proj = parity_data
        ref_s = pd.array(ref["low_since"], dtype="Int32")  # type: ignore[arg-type]
        proj_s = proj["ms_low_since"]
        pd.testing.assert_extension_array_equal(ref_s, proj_s.array)  # type: ignore[arg-type]

    def test_pullback_length(self, parity_data: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        ref, proj = parity_data
        ref_s = pd.array(ref["pullback_length"], dtype="Int32")  # type: ignore[arg-type]
        proj_s = proj["ms_pullback_length"]
        pd.testing.assert_extension_array_equal(ref_s, proj_s.array)  # type: ignore[arg-type]

    def test_pullback_correction_factor(
        self, parity_data: tuple[pd.DataFrame, pd.DataFrame]
    ) -> None:
        ref, proj = parity_data
        ref_vals = ref["pullback_correction_factor"].to_numpy(dtype=float)
        proj_vals = proj["ms_pullback_correction_factor"].to_numpy(dtype=float)
        np.testing.assert_array_equal(ref_vals, proj_vals)

    def test_wave_length(self, parity_data: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        ref, proj = parity_data
        ref_s = pd.array(ref["wave_length"], dtype="Int32")  # type: ignore[arg-type]
        proj_s = proj["ms_wave_length"]
        pd.testing.assert_extension_array_equal(ref_s, proj_s.array)  # type: ignore[arg-type]

    def test_pullback_breakout_level(self, parity_data: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        ref, proj = parity_data
        ref_vals = ref["pullback_breakout_level"].to_numpy(dtype=float)
        proj_vals = proj["ms_pullback_breakout_level"].to_numpy(dtype=float)
        np.testing.assert_array_equal(ref_vals, proj_vals)

    def test_pullback_price_diff(self, parity_data: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        ref, proj = parity_data
        ref_vals = ref["pullback_price_diff"].to_numpy(dtype=float)
        proj_vals = proj["ms_pullback_price_diff"].to_numpy(dtype=float)
        np.testing.assert_array_equal(ref_vals, proj_vals)

    def test_bearish_divergence(self, parity_data: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        ref, proj = parity_data
        ref_s = pd.array(ref["bearish_divergence"], dtype="boolean")  # type: ignore[arg-type]
        proj_s = proj["ms_bearish_divergence"]
        pd.testing.assert_extension_array_equal(ref_s, proj_s.array)  # type: ignore[arg-type]

    def test_bullish_divergence(self, parity_data: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        ref, proj = parity_data
        ref_s = pd.array(ref["bullish_divergence"], dtype="boolean")  # type: ignore[arg-type]
        proj_s = proj["ms_bullish_divergence"]
        pd.testing.assert_extension_array_equal(ref_s, proj_s.array)  # type: ignore[arg-type]

    def test_wave_count(self, parity_data: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        ref, proj = parity_data
        ref_s = pd.array(ref["wave_count"], dtype="Int32")  # type: ignore[arg-type]
        proj_s = proj["ms_wave_count"]
        pd.testing.assert_extension_array_equal(ref_s, proj_s.array)  # type: ignore[arg-type]

    def test_forming_wave_high(self, parity_data: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        ref, proj = parity_data
        ref_vals = ref["forming_wave_high"].to_numpy(dtype=float)
        proj_vals = proj["ms_forming_wave_high"].to_numpy(dtype=float)
        np.testing.assert_array_equal(ref_vals, proj_vals)

    def test_forming_wave_low(self, parity_data: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        ref, proj = parity_data
        ref_vals = ref["forming_wave_low"].to_numpy(dtype=float)
        proj_vals = proj["ms_forming_wave_low"].to_numpy(dtype=float)
        np.testing.assert_array_equal(ref_vals, proj_vals)

    def test_support_zone_low(self, parity_data: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        ref, proj = parity_data
        ref_vals = ref["support_zone_low"].to_numpy(dtype=float)
        proj_vals = proj["ms_support_zone_low"].to_numpy(dtype=float)
        np.testing.assert_array_equal(ref_vals, proj_vals)

    def test_support_zone_high(self, parity_data: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        ref, proj = parity_data
        ref_vals = ref["support_zone_high"].to_numpy(dtype=float)
        proj_vals = proj["ms_support_zone_high"].to_numpy(dtype=float)
        np.testing.assert_array_equal(ref_vals, proj_vals)

    def test_resistance_zone_low(self, parity_data: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        ref, proj = parity_data
        ref_vals = ref["resistance_zone_low"].to_numpy(dtype=float)
        proj_vals = proj["ms_resistance_zone_low"].to_numpy(dtype=float)
        np.testing.assert_array_equal(ref_vals, proj_vals)

    def test_resistance_zone_high(self, parity_data: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        ref, proj = parity_data
        ref_vals = ref["resistance_zone_high"].to_numpy(dtype=float)
        proj_vals = proj["ms_resistance_zone_high"].to_numpy(dtype=float)
        np.testing.assert_array_equal(ref_vals, proj_vals)

    def test_support_zone_wick_low(self, parity_data: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        ref, proj = parity_data
        ref_vals = ref["support_zone_wick_low"].to_numpy(dtype=float)
        proj_vals = proj["ms_support_zone_wick_low"].to_numpy(dtype=float)
        np.testing.assert_array_equal(ref_vals, proj_vals)

    def test_support_zone_wick_high(self, parity_data: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        ref, proj = parity_data
        ref_vals = ref["support_zone_wick_high"].to_numpy(dtype=float)
        proj_vals = proj["ms_support_zone_wick_high"].to_numpy(dtype=float)
        np.testing.assert_array_equal(ref_vals, proj_vals)

    def test_resistance_zone_wick_low(self, parity_data: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        ref, proj = parity_data
        ref_vals = ref["resistance_zone_wick_low"].to_numpy(dtype=float)
        proj_vals = proj["ms_resistance_zone_wick_low"].to_numpy(dtype=float)
        np.testing.assert_array_equal(ref_vals, proj_vals)

    def test_resistance_zone_wick_high(
        self, parity_data: tuple[pd.DataFrame, pd.DataFrame]
    ) -> None:
        ref, proj = parity_data
        ref_vals = ref["resistance_zone_wick_high"].to_numpy(dtype=float)
        proj_vals = proj["ms_resistance_zone_wick_high"].to_numpy(dtype=float)
        np.testing.assert_array_equal(ref_vals, proj_vals)

    def test_support_is_double(self, parity_data: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        ref, proj = parity_data
        ref_s = pd.array(ref["support_is_double"], dtype="boolean")  # type: ignore[arg-type]
        proj_s = proj["ms_support_is_double"]
        pd.testing.assert_extension_array_equal(ref_s, proj_s.array)  # type: ignore[arg-type]

    def test_resistance_is_double(self, parity_data: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        ref, proj = parity_data
        ref_s = pd.array(ref["resistance_is_double"], dtype="boolean")  # type: ignore[arg-type]
        proj_s = proj["ms_resistance_is_double"]
        pd.testing.assert_extension_array_equal(ref_s, proj_s.array)  # type: ignore[arg-type]

    def test_support_overlap_count(self, parity_data: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        ref, proj = parity_data
        ref_s = pd.array(ref["support_overlap_count"], dtype="Int32")  # type: ignore[arg-type]
        proj_s = proj["ms_support_overlap_count"]
        pd.testing.assert_extension_array_equal(ref_s, proj_s.array)  # type: ignore[arg-type]

    def test_resistance_overlap_count(self, parity_data: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        ref, proj = parity_data
        ref_s = pd.array(ref["resistance_overlap_count"], dtype="Int32")  # type: ignore[arg-type]
        proj_s = proj["ms_resistance_overlap_count"]
        pd.testing.assert_extension_array_equal(ref_s, proj_s.array)  # type: ignore[arg-type]

    # ATR, structure, distance, SFP parity tests

    def test_atr(self, parity_data: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        ref, proj = parity_data
        ref_vals = ref["atr"].to_numpy(dtype=float)
        proj_vals = proj["ms_atr"].to_numpy(dtype=float)
        np.testing.assert_array_equal(ref_vals, proj_vals)

    def test_structure_break_level(self, parity_data: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        ref, proj = parity_data
        ref_vals = ref["structure_break_level"].to_numpy(dtype=float)
        proj_vals = proj["ms_structure_break_level"].to_numpy(dtype=float)
        np.testing.assert_array_equal(ref_vals, proj_vals)

    def test_wave_amplitude(self, parity_data: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        ref, proj = parity_data
        ref_vals = ref["wave_amplitude"].to_numpy(dtype=float)
        proj_vals = proj["ms_wave_amplitude"].to_numpy(dtype=float)
        np.testing.assert_array_equal(ref_vals, proj_vals)

    def test_wave_slope(self, parity_data: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        ref, proj = parity_data
        ref_vals = ref["wave_slope"].to_numpy(dtype=float)
        proj_vals = proj["ms_wave_slope"].to_numpy(dtype=float)
        np.testing.assert_array_equal(ref_vals, proj_vals)

    def test_distance_to_support(self, parity_data: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        ref, proj = parity_data
        ref_vals = ref["distance_to_support"].to_numpy(dtype=float)
        proj_vals = proj["ms_distance_to_support"].to_numpy(dtype=float)
        np.testing.assert_array_equal(ref_vals, proj_vals)

    def test_distance_to_resistance(self, parity_data: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        ref, proj = parity_data
        ref_vals = ref["distance_to_resistance"].to_numpy(dtype=float)
        proj_vals = proj["ms_distance_to_resistance"].to_numpy(dtype=float)
        np.testing.assert_array_equal(ref_vals, proj_vals)

    def test_bars_since_last_top(self, parity_data: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        ref, proj = parity_data
        ref_s = pd.array(ref["bars_since_last_top"], dtype="Int32")  # type: ignore[arg-type]
        proj_s = proj["ms_bars_since_last_top"]
        pd.testing.assert_extension_array_equal(ref_s, proj_s.array)  # type: ignore[arg-type]

    def test_bars_since_last_bottom(self, parity_data: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        ref, proj = parity_data
        ref_s = pd.array(ref["bars_since_last_bottom"], dtype="Int32")  # type: ignore[arg-type]
        proj_s = proj["ms_bars_since_last_bottom"]
        pd.testing.assert_extension_array_equal(ref_s, proj_s.array)  # type: ignore[arg-type]

    def test_sfp_high(self, parity_data: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        ref, proj = parity_data
        ref_s = pd.array(ref["sfp_high"], dtype="boolean")  # type: ignore[arg-type]
        proj_s = proj["ms_sfp_high"]
        pd.testing.assert_extension_array_equal(ref_s, proj_s.array)  # type: ignore[arg-type]

    def test_sfp_low(self, parity_data: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        ref, proj = parity_data
        ref_s = pd.array(ref["sfp_low"], dtype="boolean")  # type: ignore[arg-type]
        proj_s = proj["ms_sfp_low"]
        pd.testing.assert_extension_array_equal(ref_s, proj_s.array)  # type: ignore[arg-type]

    def test_structure_break_confirmed(
        self, parity_data: tuple[pd.DataFrame, pd.DataFrame]
    ) -> None:
        ref, proj = parity_data
        ref_s = pd.array(ref["structure_break_confirmed"], dtype="boolean")  # type: ignore[arg-type]
        proj_s = proj["ms_structure_break_confirmed"]
        pd.testing.assert_extension_array_equal(ref_s, proj_s.array)  # type: ignore[arg-type]

    # Zone lifecycle parity tests

    def test_zone_break_support(self, parity_data: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        ref, proj = parity_data
        ref_s = pd.array(ref["zone_break_support"], dtype="boolean")  # type: ignore[arg-type]
        proj_s = proj["ms_zone_break_support"]
        pd.testing.assert_extension_array_equal(ref_s, proj_s.array)  # type: ignore[arg-type]

    def test_zone_break_resistance(self, parity_data: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        ref, proj = parity_data
        ref_s = pd.array(ref["zone_break_resistance"], dtype="boolean")  # type: ignore[arg-type]
        proj_s = proj["ms_zone_break_resistance"]
        pd.testing.assert_extension_array_equal(ref_s, proj_s.array)  # type: ignore[arg-type]

    def test_zone_retest_support(self, parity_data: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        ref, proj = parity_data
        ref_s = pd.array(ref["zone_retest_support"], dtype="boolean")  # type: ignore[arg-type]
        proj_s = proj["ms_zone_retest_support"]
        pd.testing.assert_extension_array_equal(ref_s, proj_s.array)  # type: ignore[arg-type]

    def test_zone_retest_resistance(self, parity_data: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        ref, proj = parity_data
        ref_s = pd.array(ref["zone_retest_resistance"], dtype="boolean")  # type: ignore[arg-type]
        proj_s = proj["ms_zone_retest_resistance"]
        pd.testing.assert_extension_array_equal(ref_s, proj_s.array)  # type: ignore[arg-type]

    def test_zone_retest_count_support(
        self, parity_data: tuple[pd.DataFrame, pd.DataFrame]
    ) -> None:
        ref, proj = parity_data
        ref_s = pd.array(ref["zone_retest_count_support"], dtype="Int32")  # type: ignore[arg-type]
        proj_s = proj["ms_zone_retest_count_support"]
        pd.testing.assert_extension_array_equal(ref_s, proj_s.array)  # type: ignore[arg-type]

    def test_zone_retest_count_resistance(
        self, parity_data: tuple[pd.DataFrame, pd.DataFrame]
    ) -> None:
        ref, proj = parity_data
        ref_s = pd.array(ref["zone_retest_count_resistance"], dtype="Int32")  # type: ignore[arg-type]
        proj_s = proj["ms_zone_retest_count_resistance"]
        pd.testing.assert_extension_array_equal(ref_s, proj_s.array)  # type: ignore[arg-type]

    def test_zone_flip_support(self, parity_data: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        ref, proj = parity_data
        ref_s = pd.array(ref["zone_flip_support"], dtype="boolean")  # type: ignore[arg-type]
        proj_s = proj["ms_zone_flip_support"]
        pd.testing.assert_extension_array_equal(ref_s, proj_s.array)  # type: ignore[arg-type]

    def test_zone_flip_resistance(self, parity_data: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        ref, proj = parity_data
        ref_s = pd.array(ref["zone_flip_resistance"], dtype="boolean")  # type: ignore[arg-type]
        proj_s = proj["ms_zone_flip_resistance"]
        pd.testing.assert_extension_array_equal(ref_s, proj_s.array)  # type: ignore[arg-type]

    def test_zone_failed_retest_support(
        self, parity_data: tuple[pd.DataFrame, pd.DataFrame]
    ) -> None:
        ref, proj = parity_data
        ref_s = pd.array(ref["zone_failed_retest_support"], dtype="boolean")  # type: ignore[arg-type]
        proj_s = proj["ms_zone_failed_retest_support"]
        pd.testing.assert_extension_array_equal(ref_s, proj_s.array)  # type: ignore[arg-type]

    def test_zone_failed_retest_resistance(
        self, parity_data: tuple[pd.DataFrame, pd.DataFrame]
    ) -> None:
        ref, proj = parity_data
        ref_s = pd.array(ref["zone_failed_retest_resistance"], dtype="boolean")  # type: ignore[arg-type]
        proj_s = proj["ms_zone_failed_retest_resistance"]
        pd.testing.assert_extension_array_equal(ref_s, proj_s.array)  # type: ignore[arg-type]

    # Zone quality and pullback ATR factor parity tests

    def test_zone_quality_support(self, parity_data: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        ref, proj = parity_data
        ref_vals = ref["zone_quality_support"].to_numpy(dtype=float)
        proj_vals = proj["ms_zone_quality_support"].to_numpy(dtype=float)
        np.testing.assert_array_equal(ref_vals, proj_vals)

    def test_zone_quality_resistance(self, parity_data: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        ref, proj = parity_data
        ref_vals = ref["zone_quality_resistance"].to_numpy(dtype=float)
        proj_vals = proj["ms_zone_quality_resistance"].to_numpy(dtype=float)
        np.testing.assert_array_equal(ref_vals, proj_vals)

    def test_pullback_atr_factor(self, parity_data: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        ref, proj = parity_data
        ref_vals = ref["pullback_atr_factor"].to_numpy(dtype=float)
        proj_vals = proj["ms_pullback_atr_factor"].to_numpy(dtype=float)
        np.testing.assert_array_equal(ref_vals, proj_vals)

    # Wave volume and amplitude ratio parity tests

    def test_wave_volume(self, parity_data: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        ref, proj = parity_data
        ref_vals = ref["wave_volume"].to_numpy(dtype=float)
        proj_vals = proj["ms_wave_volume"].to_numpy(dtype=float)
        np.testing.assert_array_equal(ref_vals, proj_vals)

    def test_wave_volume_ratio(self, parity_data: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        ref, proj = parity_data
        ref_vals = ref["wave_volume_ratio"].to_numpy(dtype=float)
        proj_vals = proj["ms_wave_volume_ratio"].to_numpy(dtype=float)
        np.testing.assert_array_equal(ref_vals, proj_vals)

    def test_wave_amplitude_ratio(self, parity_data: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        ref, proj = parity_data
        ref_vals = ref["wave_amplitude_ratio"].to_numpy(dtype=float)
        proj_vals = proj["ms_wave_amplitude_ratio"].to_numpy(dtype=float)
        np.testing.assert_array_equal(ref_vals, proj_vals)

    # Trend duration and multi-wave pattern parity tests

    def test_trend_wave_count(self, parity_data: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        ref, proj = parity_data
        ref_s = pd.array(ref["trend_wave_count"], dtype="Int32")  # type: ignore[arg-type]
        proj_s = proj["ms_trend_wave_count"]
        pd.testing.assert_extension_array_equal(ref_s, proj_s.array)  # type: ignore[arg-type]

    def test_three_push_up(self, parity_data: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        ref, proj = parity_data
        ref_s = pd.array(ref["three_push_up"], dtype="boolean")  # type: ignore[arg-type]
        proj_s = proj["ms_three_push_up"]
        pd.testing.assert_extension_array_equal(ref_s, proj_s.array)  # type: ignore[arg-type]

    def test_three_push_down(self, parity_data: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        ref, proj = parity_data
        ref_s = pd.array(ref["three_push_down"], dtype="boolean")  # type: ignore[arg-type]
        proj_s = proj["ms_three_push_down"]
        pd.testing.assert_extension_array_equal(ref_s, proj_s.array)  # type: ignore[arg-type]

    def test_trend_duration(self, parity_data: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        ref, proj = parity_data
        ref_s = pd.array(ref["trend_duration"], dtype="Int32")  # type: ignore[arg-type]
        proj_s = proj["ms_trend_duration"]
        pd.testing.assert_extension_array_equal(ref_s, proj_s.array)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Column validation
# ---------------------------------------------------------------------------


class TestColumnValidation:
    def test_unknown_column_raises(self) -> None:
        df = _synthetic_df()
        store: dict[str, MarketStructureHelper] = {}
        with pytest.raises(ValueError, match="Unknown column"):
            attach_market_structure(
                df, {"pair": "TEST"}, store, hist_col="tsi_hist", columns=("bogus",)
            )

    def test_none_projects_all(self) -> None:
        df = _synthetic_df()
        store: dict[str, MarketStructureHelper] = {}
        result, _ = attach_market_structure(
            df, {"pair": "TEST"}, store, hist_col="tsi_hist", columns=None
        )
        for col in VALID_COLUMNS:
            assert f"ms_{col}" in result.columns

    def test_subset_only_requested(self) -> None:
        df = _synthetic_df()
        store: dict[str, MarketStructureHelper] = {}
        result, _ = attach_market_structure(
            df, {"pair": "TEST"}, store, hist_col="tsi_hist", columns=("wave_side",)
        )
        ms_cols = [c for c in result.columns if c.startswith("ms_")]
        assert ms_cols == ["ms_wave_side"]


# ---------------------------------------------------------------------------
# Backtest projection (synthetic)
# ---------------------------------------------------------------------------


class TestBacktestProjection:
    def test_wave_side_values(self) -> None:
        df = _synthetic_df()
        store: dict[str, MarketStructureHelper] = {}
        result, _ = attach_market_structure(
            df, {"pair": "TEST"}, store, hist_col="tsi_hist", columns=("wave_side",)
        )
        sides = list(result["ms_wave_side"])
        # Bars 0-1: up (hist >= 0), bars 2-3: down, bars 4-5: up, bar 6: down
        assert sides == ["up", "up", "down", "down", "up", "up", "down"]

    def test_wave_id_values(self) -> None:
        df = _synthetic_df()
        store: dict[str, MarketStructureHelper] = {}
        result, _ = attach_market_structure(
            df, {"pair": "TEST"}, store, hist_col="tsi_hist", columns=("wave_id",)
        )
        ids = list(result["ms_wave_id"])
        assert ids == ["w-0", "w-0", "w-1", "w-1", "w-2", "w-2", "forming-3"]

    def test_helper_stored(self) -> None:
        df = _synthetic_df()
        store: dict[str, MarketStructureHelper] = {}
        _, helper = attach_market_structure(
            df, {"pair": "TEST"}, store, hist_col="tsi_hist", columns=()
        )
        assert store["TEST"] is helper

    def test_max_waves_restored(self) -> None:
        df = _synthetic_df()
        store: dict[str, MarketStructureHelper] = {}
        _, helper = attach_market_structure(
            df, {"pair": "TEST"}, store, hist_col="tsi_hist", columns=(), max_waves=50
        )
        assert helper.max_waves == 50


# ---------------------------------------------------------------------------
# Live path
# ---------------------------------------------------------------------------


class TestLivePath:
    def test_first_call_hydrates(self) -> None:
        df = _synthetic_df()
        store: dict[str, MarketStructureHelper] = {}
        _, helper = attach_market_structure(
            df, {"pair": "TEST"}, store, hist_col="tsi_hist", columns=("wave_side",)
        )
        assert "TEST" in store
        assert helper.total_candles_registered == 7

    def test_subsequent_call_registers_candle(self) -> None:
        df = _synthetic_df()
        store: dict[str, MarketStructureHelper] = {}
        attach_market_structure(df, {"pair": "TEST"}, store, hist_col="tsi_hist", columns=())

        # Append a new bar and call again.
        df2 = pd.concat(
            [
                df,
                pd.DataFrame(
                    [
                        {
                            "open_time": 8000,
                            "open": 97.0,
                            "high": 100.0,
                            "low": 94.0,
                            "close": 95.0,
                            "volume": 1.0,
                            "tsi_hist": -0.5,
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
        _, helper = attach_market_structure(
            df2, {"pair": "TEST"}, store, hist_col="tsi_hist", columns=("wave_side",)
        )
        assert helper.total_candles_registered == 8

    def test_dedup_on_same_open_time(self) -> None:
        df = _synthetic_df()
        store: dict[str, MarketStructureHelper] = {}
        attach_market_structure(df, {"pair": "TEST"}, store, hist_col="tsi_hist", columns=())

        # Call again with the same DataFrame — dedup should prevent double-count.
        _, helper = attach_market_structure(
            df, {"pair": "TEST"}, store, hist_col="tsi_hist", columns=("wave_side",)
        )
        assert helper.total_candles_registered == 7


# ---------------------------------------------------------------------------
# Desync detection
# ---------------------------------------------------------------------------


class TestDesync:
    def test_raises_on_backward_open_time(self) -> None:
        df = _synthetic_df()
        store: dict[str, MarketStructureHelper] = {}
        attach_market_structure(df, {"pair": "TEST"}, store, hist_col="tsi_hist", columns=())

        # Older DataFrame: last open_time = 5000 < helper's 7000.
        older_df = df.iloc[:5].copy().reset_index(drop=True)
        with pytest.raises(MarketStructureDesyncError, match="older"):
            attach_market_structure(
                older_df, {"pair": "TEST"}, store, hist_col="tsi_hist", columns=()
            )

    def test_no_error_on_same_open_time(self) -> None:
        df = _synthetic_df()
        store: dict[str, MarketStructureHelper] = {}
        attach_market_structure(df, {"pair": "TEST"}, store, hist_col="tsi_hist", columns=())

        # Same DataFrame again — should not raise.
        attach_market_structure(df, {"pair": "TEST"}, store, hist_col="tsi_hist", columns=())

    def test_no_error_on_forward_open_time(self) -> None:
        df = _synthetic_df()
        store: dict[str, MarketStructureHelper] = {}
        attach_market_structure(df, {"pair": "TEST"}, store, hist_col="tsi_hist", columns=())

        # Newer DataFrame.
        newer_df = pd.concat(
            [
                df,
                pd.DataFrame(
                    [
                        {
                            "open_time": 8000,
                            "open": 97.0,
                            "high": 100.0,
                            "low": 94.0,
                            "close": 95.0,
                            "volume": 1.0,
                            "tsi_hist": -0.5,
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
        attach_market_structure(newer_df, {"pair": "TEST"}, store, hist_col="tsi_hist", columns=())


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_dataframe(self) -> None:
        df = pd.DataFrame(
            columns=["open_time", "open", "high", "low", "close", "volume", "tsi_hist"]
        )
        store: dict[str, MarketStructureHelper] = {}
        result, helper = attach_market_structure(
            df, {"pair": "TEST"}, store, hist_col="tsi_hist", columns=("wave_side",)
        )
        assert len(result) == 0
        assert helper.total_candles_registered == 0

    def test_single_candle(self) -> None:
        df = pd.DataFrame(
            [
                {
                    "open_time": 1000,
                    "open": 100.0,
                    "high": 105.0,
                    "low": 98.0,
                    "close": 103.0,
                    "volume": 1.0,
                    "tsi_hist": 0.4,
                }
            ]
        )
        store: dict[str, MarketStructureHelper] = {}
        result, _ = attach_market_structure(
            df,
            {"pair": "TEST"},
            store,
            hist_col="tsi_hist",
            columns=("wave_side", "is_trending_up"),
        )
        assert list(result["ms_wave_side"]) == ["up"]
        assert list(result["ms_is_trending_up"]) == [False]

    def test_no_flips(self) -> None:
        """All same-sign histogram — one long wave, no confirmed waves."""
        df = pd.DataFrame(
            {
                "open_time": [1000, 2000, 3000, 4000],
                "open": [100.0, 102.0, 104.0, 103.0],
                "high": [105.0, 106.0, 108.0, 107.0],
                "low": [98.0, 100.0, 102.0, 101.0],
                "close": [103.0, 105.0, 107.0, 104.0],
                "volume": [1.0] * 4,
                "tsi_hist": [0.1, 0.2, 0.3, 0.1],
            }
        )
        store: dict[str, MarketStructureHelper] = {}
        result, _ = attach_market_structure(
            df,
            {"pair": "TEST"},
            store,
            hist_col="tsi_hist",
            columns=("wave_side", "last_top_price"),
        )
        assert all(s == "up" for s in result["ms_wave_side"])
        assert all(np.isnan(v) for v in result["ms_last_top_price"])
