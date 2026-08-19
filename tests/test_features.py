from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.features import (
    _forward_window_count,
    add_days_since_last_repair,
    calculate_priority_components,
    risk_level_from_percentile,
)


class RepairFeatureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.panel = pd.DataFrame(
            {
                "grid_id": ["a", "a", "b"],
                "date": pd.to_datetime(["2025-01-01", "2025-01-10", "2025-01-10"]),
                "past_potholes_90d": [0, 1, 0],
                "past_potholes_total": [0, 2, 0],
            }
        )

    def test_no_repair_history_is_flag_zero_and_days_zero(self) -> None:
        empty = pd.DataFrame(columns=["grid_id", "repair_date"])
        result = add_days_since_last_repair(self.panel, empty)
        self.assertTrue(result["has_repair_history"].eq(0).all())
        self.assertTrue(result["days_since_last_repair"].eq(0).all())

    def test_repair_history_uses_only_past_repairs(self) -> None:
        repairs = pd.DataFrame(
            {"grid_id": ["a"], "repair_date": pd.to_datetime(["2025-01-05"])}
        )
        result = add_days_since_last_repair(self.panel, repairs).sort_values(["grid_id", "date"])
        grid_a = result[result["grid_id"].eq("a")]
        self.assertEqual(grid_a["has_repair_history"].tolist(), [0, 1])
        self.assertEqual(grid_a["days_since_last_repair"].tolist(), [0, 5])
        grid_b = result[result["grid_id"].eq("b")]
        self.assertEqual(grid_b["has_repair_history"].iloc[0], 0)
        self.assertEqual(grid_b["days_since_last_repair"].iloc[0], 0)


class TargetAndPriorityTests(unittest.TestCase):
    def test_forward_target_excludes_current_day_and_incomplete_tail(self) -> None:
        counts = pd.Series([0, 1, 0, 1])
        result = _forward_window_count(counts, horizon_days=2)
        self.assertEqual(result.iloc[:2].tolist(), [1.0, 1.0])
        self.assertTrue(result.iloc[2:].isna().all())

    def test_priority_weights_are_renormalized_without_importance(self) -> None:
        frame = pd.DataFrame(
            {"past_potholes_90d": [5.0], "past_potholes_total": [10.0]}
        )
        result = calculate_priority_components(
            frame,
            pd.Series([0.6]),
            {"past_potholes_90d": 10.0, "past_potholes_total": 20.0},
            {},
        ).iloc[0]
        self.assertAlmostEqual(result["recurrence_score"], 0.5)
        self.assertAlmostEqual(result["priority_weight_risk"], 0.75 / 0.90)
        self.assertAlmostEqual(result["priority_weight_recurrence"], 0.15 / 0.90)
        self.assertEqual(result["priority_weight_importance"], 0.0)
        self.assertAlmostEqual(result["priority_score"], (0.75 * 0.6 + 0.15 * 0.5) / 0.90)

    def test_priority_uses_importance_when_available(self) -> None:
        frame = pd.DataFrame(
            {
                "past_potholes_90d": [0.0],
                "past_potholes_total": [0.0],
                "traffic_volume": [800.0],
            }
        )
        result = calculate_priority_components(
            frame,
            pd.Series([0.4]),
            {"past_potholes_90d": 1.0, "past_potholes_total": 1.0},
            {"traffic_volume": 1000.0},
        ).iloc[0]
        self.assertAlmostEqual(result["importance_score"], 0.8)
        self.assertAlmostEqual(result["priority_score"], 0.75 * 0.4 + 0.10 * 0.8)
        self.assertAlmostEqual(
            result["priority_weight_risk"]
            + result["priority_weight_recurrence"]
            + result["priority_weight_importance"],
            1.0,
        )

    def test_risk_levels_are_relative_percentiles(self) -> None:
        values = [risk_level_from_percentile(x) for x in [0.50, 0.51, 0.81, 0.96]]
        self.assertEqual(values, ["낮음", "보통", "높음", "매우 높음"])


if __name__ == "__main__":
    unittest.main()
