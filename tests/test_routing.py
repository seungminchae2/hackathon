from __future__ import annotations

import unittest

import pandas as pd

from src.routing import analyze_route_risk, choose_safe_route


class RouteRiskTests(unittest.TestCase):
    def setUp(self) -> None:
        self.predictions = pd.DataFrame(
            {
                "grid_id": ["danger", "safe"],
                "grid_lat": [35.8200, 35.8500],
                "grid_lon": [127.1000, 127.1000],
                "risk_score": [0.9, 0.1],
                "risk_level": ["매우 높음", "낮음"],
            }
        )

    def test_route_detects_nearby_high_risk_grid(self) -> None:
        route = {
            "path": [[127.095, 35.820], [127.105, 35.820]],
            "distance_m": 1000,
        }
        analyzed = analyze_route_risk(route, self.predictions)
        self.assertEqual(analyzed["high_risk_count"], 1)
        self.assertIn("danger", analyzed["danger_grid_ids"])
        self.assertGreater(analyzed["risk_exposure"], 0)

    def test_safe_detour_is_selected_within_time_limit(self) -> None:
        routes = [
            {"duration_s": 600, "risk_exposure": 0.8, "high_risk_count": 3},
            {"duration_s": 690, "risk_exposure": 0.2, "high_risk_count": 1},
            {"duration_s": 1000, "risk_exposure": 0.0, "high_risk_count": 0},
        ]
        selected, message = choose_safe_route(routes)
        self.assertEqual(selected, 1)
        self.assertIn("우회 경로", message)

    def test_no_danger_keeps_baseline(self) -> None:
        selected, _ = choose_safe_route(
            [{"duration_s": 600, "risk_exposure": 0.0, "high_risk_count": 0}]
        )
        self.assertEqual(selected, 0)


if __name__ == "__main__":
    unittest.main()
