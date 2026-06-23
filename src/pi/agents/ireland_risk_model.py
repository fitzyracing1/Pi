"""Ireland-specific deterministic risk model agent.

The model is intentionally transparent: every score is based on named input
features, bounded to a 0-100 range, and returned with the drivers that moved
the score. It is designed as a starting point for Irish property and business
risk triage, not as a replacement for underwriting judgement or regulated
advice.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


REPUBLIC_OF_IRELAND_COUNTIES = {
    "CARLOW",
    "CAVAN",
    "CLARE",
    "CORK",
    "DONEGAL",
    "DUBLIN",
    "GALWAY",
    "KERRY",
    "KILDARE",
    "KILKENNY",
    "LAOIS",
    "LEITRIM",
    "LIMERICK",
    "LONGFORD",
    "LOUTH",
    "MAYO",
    "MEATH",
    "MONAGHAN",
    "OFFALY",
    "ROSCOMMON",
    "SLIGO",
    "TIPPERARY",
    "WATERFORD",
    "WESTMEATH",
    "WEXFORD",
    "WICKLOW",
}

NORTHERN_IRELAND_COUNTIES = {
    "ANTRIM",
    "ARMAGH",
    "DERRY",
    "DOWN",
    "FERMANAGH",
    "TYRONE",
}

SUPPORTED_COUNTIES = REPUBLIC_OF_IRELAND_COUNTIES | NORTHERN_IRELAND_COUNTIES

COUNTY_ALIASES = {
    "LONDONDERRY": "DERRY",
    "CO DERRY": "DERRY",
    "CO LONDONDERRY": "DERRY",
}

ATLANTIC_EXPOSED_COUNTIES = {
    "CLARE",
    "CORK",
    "DONEGAL",
    "GALWAY",
    "KERRY",
    "MAYO",
    "SLIGO",
}

COASTAL_COUNTIES = {
    "ANTRIM",
    "CLARE",
    "CORK",
    "DERRY",
    "DONEGAL",
    "DOWN",
    "DUBLIN",
    "GALWAY",
    "KERRY",
    "LOUTH",
    "MAYO",
    "SLIGO",
    "WATERFORD",
    "WEXFORD",
    "WICKLOW",
}

COUNTY_FLOOD_MULTIPLIER = {
    "CORK": 1.16,
    "DUBLIN": 1.12,
    "GALWAY": 1.14,
    "KERRY": 1.10,
    "LIMERICK": 1.12,
    "MAYO": 1.08,
    "OFFALY": 1.08,
    "ROSCOMMON": 1.08,
    "TIPPERARY": 1.08,
    "WATERFORD": 1.10,
    "WEXFORD": 1.08,
}

URBAN_BUSINESS_COUNTIES = {
    "ANTRIM",
    "CORK",
    "DUBLIN",
    "GALWAY",
    "LIMERICK",
    "WATERFORD",
}

DIMENSION_WEIGHTS = {
    "flood": 0.25,
    "storm": 0.20,
    "property_condition": 0.18,
    "liability": 0.14,
    "business_interruption": 0.13,
    "cyber": 0.10,
}


@dataclass(frozen=True)
class IrelandRiskProfile:
    """Input facts for evaluating an Irish property or business risk."""

    county: str
    sector: str
    asset_type: str = "commercial_property"
    sum_insured_eur: float = 0.0
    flood_zone: str = "unknown"
    coastal_exposure: bool = False
    wind_exposure: str = "moderate"
    construction_year: int | None = None
    ber_rating: str = "unknown"
    occupancy: str = "owner_occupied"
    claims_count_5y: int = 0
    has_flood_defences: bool = False
    business_continuity_plan: bool = False
    public_liability_exposure: str = "moderate"
    cyber_maturity: str = "baseline"


@dataclass(frozen=True)
class RiskDimensionScore:
    """A score for one risk dimension and the drivers behind it."""

    name: str
    score: float
    drivers: tuple[str, ...]


@dataclass(frozen=True)
class IrelandRiskAssessment:
    """Agent output containing overall and dimension-level risk scores."""

    overall_score: float
    band: str
    county: str
    jurisdiction: str
    dimensions: dict[str, RiskDimensionScore]
    recommendations: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation of the assessment."""

        return asdict(self)


class IrelandRiskModelAgent:
    """Risk model agent tuned for Ireland-specific underwriting triage."""

    def evaluate(self, profile: IrelandRiskProfile) -> IrelandRiskAssessment:
        """Evaluate a profile and return a complete risk assessment."""

        county = self._normalise_county(profile.county)
        self._validate_profile(profile, county)

        flood = self._score_flood(profile, county)
        storm = self._score_storm(profile, county)
        property_condition = self._score_property_condition(profile)
        liability = self._score_liability(profile, county)
        business_interruption = self._score_business_interruption(
            profile,
            county,
            flood.score,
            storm.score,
        )
        cyber = self._score_cyber(profile, county)

        dimensions = {
            "flood": flood,
            "storm": storm,
            "property_condition": property_condition,
            "liability": liability,
            "business_interruption": business_interruption,
            "cyber": cyber,
        }

        overall_score = round(
            sum(
                dimensions[name].score * weight
                for name, weight in DIMENSION_WEIGHTS.items()
            ),
            2,
        )

        return IrelandRiskAssessment(
            overall_score=overall_score,
            band=self._band(overall_score),
            county=county.title(),
            jurisdiction=self._jurisdiction_for_county(county),
            dimensions=dimensions,
            recommendations=self._recommendations(dimensions),
        )

    def evaluate_many(
        self,
        profiles: list[IrelandRiskProfile],
    ) -> list[IrelandRiskAssessment]:
        """Evaluate several profiles with the same Ireland model."""

        return [self.evaluate(profile) for profile in profiles]

    def _score_flood(
        self,
        profile: IrelandRiskProfile,
        county: str,
    ) -> RiskDimensionScore:
        zone_scores = {
            "low": 12.0,
            "moderate": 40.0,
            "medium": 40.0,
            "high": 72.0,
            "unknown": 32.0,
        }
        drivers: list[str] = []
        flood_zone = _normalise_value(profile.flood_zone)
        score = zone_scores.get(flood_zone, zone_scores["unknown"])
        drivers.append(f"flood zone set to {flood_zone or 'unknown'}")

        multiplier = COUNTY_FLOOD_MULTIPLIER.get(county, 1.0)
        if multiplier > 1.0:
            score *= multiplier
            drivers.append(f"{county.title()} county flood multiplier applied")

        if profile.coastal_exposure:
            score += 8.0
            drivers.append("coastal exposure increases flood surge risk")
        elif county in COASTAL_COUNTIES:
            score += 3.0
            drivers.append("county has coastal exposure")

        if profile.has_flood_defences:
            score -= 12.0
            drivers.append("declared flood defences reduce expected loss")

        claims_adjustment = min(profile.claims_count_5y * 5.0, 15.0)
        if claims_adjustment:
            score += claims_adjustment
            drivers.append("recent claims history increases flood concern")

        if profile.sum_insured_eur >= 2_000_000:
            score += 5.0
            drivers.append("large declared value increases severity")

        return RiskDimensionScore(
            name="flood",
            score=_round_score(score),
            drivers=tuple(drivers),
        )

    def _score_storm(
        self,
        profile: IrelandRiskProfile,
        county: str,
    ) -> RiskDimensionScore:
        wind_scores = {
            "low": 18.0,
            "moderate": 35.0,
            "medium": 35.0,
            "high": 58.0,
            "unknown": 35.0,
        }
        drivers: list[str] = []
        wind_exposure = _normalise_value(profile.wind_exposure)
        score = wind_scores.get(wind_exposure, wind_scores["unknown"])
        drivers.append(f"wind exposure set to {wind_exposure or 'unknown'}")

        if county in ATLANTIC_EXPOSED_COUNTIES:
            score *= 1.20
            drivers.append("Atlantic county exposure increases storm risk")
        elif county in COASTAL_COUNTIES:
            score *= 1.08
            drivers.append("coastal county exposure increases wind risk")

        if profile.coastal_exposure:
            score += 10.0
            drivers.append("asset-level coastal exposure increases storm surge risk")

        if profile.construction_year is not None:
            if profile.construction_year < 1980:
                score += 8.0
                drivers.append("pre-1980 construction increases vulnerability")
            elif profile.construction_year >= 2010:
                score -= 4.0
                drivers.append("newer construction reduces storm vulnerability")

        return RiskDimensionScore(
            name="storm",
            score=_round_score(score),
            drivers=tuple(drivers),
        )

    def _score_property_condition(
        self,
        profile: IrelandRiskProfile,
    ) -> RiskDimensionScore:
        drivers: list[str] = []
        score = 30.0

        if profile.construction_year is None:
            score += 5.0
            drivers.append("construction year unknown")
        elif profile.construction_year < 1950:
            score += 18.0
            drivers.append("pre-1950 construction increases condition risk")
        elif profile.construction_year < 1980:
            score += 10.0
            drivers.append("older construction increases condition risk")
        elif profile.construction_year >= 2010:
            score -= 5.0
            drivers.append("newer construction lowers condition risk")

        ber_rating = _normalise_value(profile.ber_rating).upper()
        if ber_rating.startswith("A"):
            score -= 8.0
            drivers.append("BER A rating lowers energy and condition risk")
        elif ber_rating.startswith("B"):
            score -= 4.0
            drivers.append("BER B rating lowers energy and condition risk")
        elif ber_rating.startswith(("E", "F", "G")):
            score += 14.0
            drivers.append("low BER rating increases retrofit and condition risk")
        elif ber_rating in {"UNKNOWN", ""}:
            score += 4.0
            drivers.append("BER rating unknown")

        occupancy = _normalise_value(profile.occupancy)
        if occupancy in {"vacant", "unoccupied"}:
            score += 12.0
            drivers.append("vacancy increases property condition risk")
        elif occupancy in {"tenant", "leased"}:
            score += 5.0
            drivers.append("tenant occupancy increases maintenance uncertainty")

        return RiskDimensionScore(
            name="property_condition",
            score=_round_score(score),
            drivers=tuple(drivers),
        )

    def _score_liability(
        self,
        profile: IrelandRiskProfile,
        county: str,
    ) -> RiskDimensionScore:
        exposure_scores = {
            "low": 18.0,
            "moderate": 35.0,
            "medium": 35.0,
            "high": 62.0,
            "unknown": 38.0,
        }
        drivers: list[str] = []
        exposure = _normalise_value(profile.public_liability_exposure)
        score = exposure_scores.get(exposure, exposure_scores["unknown"])
        drivers.append(f"public liability exposure set to {exposure or 'unknown'}")

        sector = _normalise_value(profile.sector)
        if sector in {"construction", "hospitality", "leisure", "retail"}:
            score += 10.0
            drivers.append(f"{sector} sector has elevated public interaction risk")
        elif sector in {"agriculture", "manufacturing", "food_production"}:
            score += 6.0
            drivers.append(f"{sector} sector has operational injury exposure")

        if county in URBAN_BUSINESS_COUNTIES:
            score += 4.0
            drivers.append("urban business county increases footfall exposure")

        claims_adjustment = min(profile.claims_count_5y * 4.0, 12.0)
        if claims_adjustment:
            score += claims_adjustment
            drivers.append("recent claims history increases liability concern")

        return RiskDimensionScore(
            name="liability",
            score=_round_score(score),
            drivers=tuple(drivers),
        )

    def _score_business_interruption(
        self,
        profile: IrelandRiskProfile,
        county: str,
        flood_score: float,
        storm_score: float,
    ) -> RiskDimensionScore:
        drivers: list[str] = []
        sector = _normalise_value(profile.sector)
        score = 28.0

        if sector in {
            "food_production",
            "hospitality",
            "logistics",
            "manufacturing",
            "retail",
            "tourism",
        }:
            score += 12.0
            drivers.append(f"{sector} sector depends on premises or supply chains")
        elif sector in {"professional_services", "technology"}:
            score += 4.0
            drivers.append(f"{sector} sector has lower premises dependency")

        climate_adjustment = (flood_score * 0.14) + (storm_score * 0.10)
        score += climate_adjustment
        drivers.append("flood and storm scores feed business interruption risk")

        if county in ATLANTIC_EXPOSED_COUNTIES:
            score += 4.0
            drivers.append("Atlantic county exposure can disrupt transport links")

        if profile.business_continuity_plan:
            score -= 14.0
            drivers.append("business continuity plan lowers interruption risk")

        if profile.sum_insured_eur >= 2_000_000:
            score += 5.0
            drivers.append("large declared value increases interruption severity")

        return RiskDimensionScore(
            name="business_interruption",
            score=_round_score(score),
            drivers=tuple(drivers),
        )

    def _score_cyber(
        self,
        profile: IrelandRiskProfile,
        county: str,
    ) -> RiskDimensionScore:
        drivers: list[str] = []
        sector = _normalise_value(profile.sector)
        score = 24.0

        if sector in {"financial_services", "healthcare", "technology"}:
            score += 16.0
            drivers.append(f"{sector} sector has high data dependency")
        elif sector in {"professional_services", "retail"}:
            score += 10.0
            drivers.append(f"{sector} sector has customer data exposure")
        elif sector in {"hospitality", "tourism"}:
            score += 6.0
            drivers.append(f"{sector} sector has payment data exposure")

        maturity = _normalise_value(profile.cyber_maturity)
        if maturity in {"strong", "mature", "iso27001"}:
            score -= 16.0
            drivers.append("strong cyber maturity lowers risk")
        elif maturity in {"weak", "low"}:
            score += 18.0
            drivers.append("weak cyber maturity increases risk")
        elif maturity in {"unknown", ""}:
            score += 8.0
            drivers.append("cyber maturity unknown")

        if county in URBAN_BUSINESS_COUNTIES:
            score += 3.0
            drivers.append("urban business concentration increases target profile")

        return RiskDimensionScore(
            name="cyber",
            score=_round_score(score),
            drivers=tuple(drivers),
        )

    def _recommendations(
        self,
        dimensions: dict[str, RiskDimensionScore],
    ) -> tuple[str, ...]:
        recommendations: list[str] = []

        if dimensions["flood"].score >= 55:
            recommendations.append(
                "Request site-level flood mapping, drainage evidence, and OPW/local "
                "authority flood defence details."
            )
        if dimensions["storm"].score >= 55:
            recommendations.append(
                "Review roof condition, windstorm protections, coastal exposure, and "
                "storm maintenance records."
            )
        if dimensions["property_condition"].score >= 50:
            recommendations.append(
                "Obtain a recent building survey, BER certificate, and planned "
                "maintenance schedule."
            )
        if dimensions["liability"].score >= 55:
            recommendations.append(
                "Validate public liability controls, incident logs, safety statements, "
                "and contractor management."
            )
        if dimensions["business_interruption"].score >= 55:
            recommendations.append(
                "Assess supplier resilience, alternative premises options, backup "
                "utilities, and business continuity testing."
            )
        if dimensions["cyber"].score >= 50:
            recommendations.append(
                "Confirm MFA, offline backups, endpoint protection, patch cadence, and "
                "incident response ownership."
            )
        if not recommendations:
            recommendations.append(
                "Maintain current controls and refresh local risk evidence annually."
            )

        return tuple(recommendations)

    def _validate_profile(
        self,
        profile: IrelandRiskProfile,
        county: str,
    ) -> None:
        if county not in SUPPORTED_COUNTIES:
            supported = ", ".join(sorted(name.title() for name in SUPPORTED_COUNTIES))
            raise ValueError(
                f"Unsupported Ireland county '{profile.county}'. "
                f"Supported counties: {supported}."
            )
        if profile.sum_insured_eur < 0:
            raise ValueError("sum_insured_eur must be greater than or equal to 0.")
        if profile.claims_count_5y < 0:
            raise ValueError("claims_count_5y must be greater than or equal to 0.")

    def _normalise_county(self, county: str) -> str:
        value = county.strip().replace(".", "").upper()
        if value.startswith("COUNTY "):
            value = value.removeprefix("COUNTY ").strip()
        elif value.startswith("CO "):
            value = value.removeprefix("CO ").strip()
        return COUNTY_ALIASES.get(value, value)

    def _jurisdiction_for_county(self, county: str) -> str:
        if county in NORTHERN_IRELAND_COUNTIES:
            return "Northern Ireland"
        return "Republic of Ireland"

    def _band(self, score: float) -> str:
        if score < 30:
            return "low"
        if score < 55:
            return "moderate"
        if score < 75:
            return "high"
        return "severe"


def _normalise_value(value: str) -> str:
    return value.strip().lower().replace(" ", "_").replace("-", "_")


def _round_score(score: float) -> float:
    return round(max(0.0, min(100.0, score)), 2)
