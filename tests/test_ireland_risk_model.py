import unittest

from pi.agents import IrelandRiskModelAgent, IrelandRiskProfile


class IrelandRiskModelAgentTest(unittest.TestCase):
    def setUp(self) -> None:
        self.agent = IrelandRiskModelAgent()

    def test_high_risk_coastal_cork_profile_scores_high(self) -> None:
        profile = IrelandRiskProfile(
            county="Co. Cork",
            sector="hospitality",
            sum_insured_eur=3_000_000,
            flood_zone="high",
            coastal_exposure=True,
            wind_exposure="high",
            construction_year=1975,
            ber_rating="F",
            occupancy="tenant",
            claims_count_5y=2,
            public_liability_exposure="high",
            cyber_maturity="weak",
        )

        assessment = self.agent.evaluate(profile)

        self.assertEqual(assessment.county, "Cork")
        self.assertEqual(assessment.jurisdiction, "Republic of Ireland")
        self.assertEqual(assessment.band, "high")
        self.assertGreaterEqual(assessment.overall_score, 55)
        self.assertGreaterEqual(assessment.dimensions["flood"].score, 90)
        self.assertGreaterEqual(assessment.dimensions["storm"].score, 75)
        self.assertTrue(
            any("flood mapping" in item for item in assessment.recommendations)
        )

    def test_controls_reduce_flood_and_business_interruption_scores(self) -> None:
        base_profile = IrelandRiskProfile(
            county="Galway",
            sector="manufacturing",
            sum_insured_eur=1_500_000,
            flood_zone="high",
            wind_exposure="moderate",
            construction_year=2015,
        )
        controlled_profile = IrelandRiskProfile(
            county="Galway",
            sector="manufacturing",
            sum_insured_eur=1_500_000,
            flood_zone="high",
            wind_exposure="moderate",
            construction_year=2015,
            has_flood_defences=True,
            business_continuity_plan=True,
        )

        base = self.agent.evaluate(base_profile)
        controlled = self.agent.evaluate(controlled_profile)

        self.assertLess(
            controlled.dimensions["flood"].score,
            base.dimensions["flood"].score,
        )
        self.assertLess(
            controlled.dimensions["business_interruption"].score,
            base.dimensions["business_interruption"].score,
        )

    def test_londonderry_alias_maps_to_derry_northern_ireland(self) -> None:
        profile = IrelandRiskProfile(
            county="County Londonderry",
            sector="retail",
        )

        assessment = self.agent.evaluate(profile)

        self.assertEqual(assessment.county, "Derry")
        self.assertEqual(assessment.jurisdiction, "Northern Ireland")

    def test_invalid_county_raises_clear_error(self) -> None:
        profile = IrelandRiskProfile(
            county="Atlantis",
            sector="retail",
        )

        with self.assertRaisesRegex(ValueError, "Unsupported Ireland county"):
            self.agent.evaluate(profile)

    def test_to_dict_returns_serialisable_shape(self) -> None:
        profile = IrelandRiskProfile(
            county="Dublin",
            sector="technology",
            cyber_maturity="mature",
            business_continuity_plan=True,
        )

        assessment_dict = self.agent.evaluate(profile).to_dict()

        self.assertEqual(assessment_dict["county"], "Dublin")
        self.assertIn("cyber", assessment_dict["dimensions"])
        self.assertIn("overall_score", assessment_dict)


if __name__ == "__main__":
    unittest.main()
