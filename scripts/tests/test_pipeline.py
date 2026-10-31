from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

import pipeline


def make_stats(overrides=None, none=False):
    """Fake Earth Engine output: every band has mean 100, std 10, median 90."""
    stats = {}
    for b in pipeline.BANDS:
        stats[f"{b}_mean"] = None if none else 100.0
        stats[f"{b}_stdDev"] = None if none else 10.0
        stats[f"{b}_median"] = None if none else 90.0
    stats.update(overrides or {})
    return stats


def fake_image(stats):
    image = MagicMock()
    image.updateMask.return_value.select.return_value.reduceRegion.return_value.getInfo.return_value = stats
    return image


REFERENCE = pd.Series([1.0, 2.0, 3.0, 4.0])


@pytest.mark.parametrize("fnu, expected", [
    (0.5, "Clean"), (5, "Clean"), (5.1, "Moderate"),
    (25, "Moderate"), (25.1, "Polluted"), (400, "Polluted"),
])
def test_status_thresholds(fnu, expected):
    assert pipeline.turbidity_risk_score(fnu, REFERENCE)["status"] == expected


def test_percentile_score():
    assert pipeline.turbidity_risk_score(3.0, REFERENCE)["risk_score"] == 50.0
    assert pipeline.turbidity_risk_score(10.0, REFERENCE)["risk_score"] == 100.0
    assert pipeline.turbidity_risk_score(0.1, REFERENCE)["risk_score"] == 0.0


@patch("pipeline.ee")
def test_feature_count_and_ratios(_ee):
    stats = make_stats({"B3_mean": 300.0, "B4_mean": 200.0, "B8_mean": 400.0})
    f = pipeline.extract_reflectance_features(fake_image(stats), MagicMock(), MagicMock())

    assert len(f) == 47                       # 11 bands x 4 + 3 ratios
    assert f["B03_center"] == f["B03_buf250_mean"] == 300.0
    assert f["ndti"] == pytest.approx(-0.2)   # (200 - 300) / (200 + 300)
    assert f["green_red_ratio"] == pytest.approx(1.5)
    assert f["nir_red_ratio"] == pytest.approx(2.0)


@patch("pipeline.ee")
def test_no_water_raises(_ee):
    with pytest.raises(ValueError):
        pipeline.extract_reflectance_features(fake_image(make_stats(none=True)), MagicMock(), MagicMock())