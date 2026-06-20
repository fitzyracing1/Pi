# Pi

Pi contains risk modelling utilities.

## Ireland Risk Model Agent

`IrelandRiskModelAgent` is a deterministic agent for Ireland-specific property
and business risk triage. It scores a submitted `IrelandRiskProfile` across six
dimensions:

- flood
- storm
- property condition
- public liability
- business interruption
- cyber

The agent returns a 0-100 overall score, a risk band (`low`, `moderate`,
`high`, or `severe`), dimension-level drivers, jurisdiction, and recommended
follow-up actions.

### Example

```python
from pi.agents import IrelandRiskModelAgent, IrelandRiskProfile

agent = IrelandRiskModelAgent()

assessment = agent.evaluate(
    IrelandRiskProfile(
        county="Co. Cork",
        sector="hospitality",
        sum_insured_eur=3_000_000,
        flood_zone="high",
        coastal_exposure=True,
        wind_exposure="high",
        construction_year=1975,
        ber_rating="F",
        claims_count_5y=2,
        public_liability_exposure="high",
        cyber_maturity="weak",
    )
)

print(assessment.overall_score)
print(assessment.band)
print(assessment.to_dict())
```

### Model scope

- Supports all Republic of Ireland counties and Northern Ireland counties.
- Normalizes common county inputs such as `Co. Cork`, `County Galway`, and
  `County Londonderry`.
- Uses Ireland-specific geographic context for Atlantic exposure, coastal
  counties, urban business concentration, and county-level flood multipliers.
- Uses transparent rules only; there are no external API calls or model
  dependencies.

This model is intended for triage and workflow routing. It should be calibrated
with verified claims, geospatial, and underwriting data before being used for
pricing, regulated advice, or automated acceptance decisions.

### Development

Run the test suite with:

```bash
PYTHONPATH=src python -m unittest
```
