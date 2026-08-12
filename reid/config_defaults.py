"""Shared lightweight configuration constants."""

DEFAULT_MODEL_TYPE = "megadescriptor-l"

SUPPORTED_MODEL_TYPES = (
    "megadescriptor-t",
    "megadescriptor-l",
    "lynx_megadescriptorV3",
    "lynx_megadescriptorV4",
    "miewid",
    "dinov2",
    "dinov3",
)


def validate_model_type(model_type: str) -> str:
    """Return a supported model identifier or raise a clear configuration error."""
    model_type = str(model_type)
    if model_type not in SUPPORTED_MODEL_TYPES:
        supported = ", ".join(SUPPORTED_MODEL_TYPES)
        raise ValueError(f"Unsupported model type '{model_type}'. Supported types: {supported}")
    return model_type
