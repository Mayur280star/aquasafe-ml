"""The nine input readings: units, physical bounds and regulatory limits.

Physical bounds are hard limits. A reading outside them cannot be real (a pH of
15, a negative concentration) and is treated as a sensor or typing error.

Guideline limits come from the WHO Guidelines for Drinking-water Quality (4th ed.)
and the Indian standard IS 10500:2012. They are shown to users next to the model
output. They are not model inputs, because the labels in this dataset do not
follow them (see the report, section 2).
"""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Feature:
    name: str
    label: str
    unit: str
    description: str
    physical_min: float
    physical_max: float
    guideline_min: float | None
    guideline_max: float | None
    guideline_source: str
    example: float


FEATURES = [
    Feature("ph", "pH", "", "Acid-base balance of the water.",
            0.0, 14.0, 6.5, 8.5, "WHO / IS 10500", 7.08),
    Feature("Hardness", "Hardness", "mg/L", "Dissolved calcium and magnesium salts, as CaCO3.",
            0.0, 2000.0, None, 600.0, "IS 10500 permissible limit", 196.4),
    Feature("Solids", "Total dissolved solids", "ppm", "All dissolved minerals and salts.",
            0.0, 100000.0, None, 2000.0, "IS 10500 permissible limit", 20927.8),
    Feature("Chloramines", "Chloramines", "ppm", "Disinfectant formed from chlorine and ammonia.",
            0.0, 50.0, None, 4.0, "WHO / US EPA", 7.13),
    Feature("Sulfate", "Sulfate", "mg/L", "Naturally occurring sulfate salts.",
            0.0, 5000.0, None, 400.0, "IS 10500 permissible limit", 333.1),
    Feature("Conductivity", "Conductivity", "µS/cm", "Electrical conductivity, a proxy for ion content.",
            0.0, 5000.0, None, 400.0, "WHO", 421.9),
    Feature("Organic_carbon", "Organic carbon", "ppm", "Total organic carbon (TOC).",
            0.0, 100.0, None, 4.0, "US EPA (source water)", 14.2),
    Feature("Trihalomethanes", "Trihalomethanes", "µg/L", "Chlorination by-products.",
            0.0, 1000.0, None, 80.0, "US EPA", 66.6),
    Feature("Turbidity", "Turbidity", "NTU", "Cloudiness from suspended particles.",
            0.0, 1000.0, None, 5.0, "WHO / IS 10500", 3.96),
]

FEATURE_NAMES = [f.name for f in FEATURES]
BY_NAME = {f.name: f for f in FEATURES}


def as_dicts():
    return [asdict(f) for f in FEATURES]


def guideline_status(name, value):
    """'ok', 'above', 'below' or 'missing' for one reading against its guideline."""
    if value is None:
        return "missing"
    feature = BY_NAME[name]
    if feature.guideline_min is not None and value < feature.guideline_min:
        return "below"
    if feature.guideline_max is not None and value > feature.guideline_max:
        return "above"
    return "ok"
