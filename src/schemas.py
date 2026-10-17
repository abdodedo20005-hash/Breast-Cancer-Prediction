from pydantic import BaseModel, ConfigDict, Field, create_model

from src.features import EXAMPLE_MALIGNANT, FEATURES

PatientFeatures = create_model(
    "PatientFeatures",
    __config__=ConfigDict(json_schema_extra={"examples": [EXAMPLE_MALIGNANT]}),
    **{name: (float, Field(..., ge=0, allow_inf_nan=False)) for name in FEATURES},
)


class PredictionResponse(BaseModel):
    label: str
    malignant_probability: float
    benign_probability: float
    threshold: float
