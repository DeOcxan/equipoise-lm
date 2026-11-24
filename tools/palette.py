"""Colours of the report figures, shared so that a model has the same colour in every figure."""

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"

COLORS = {
    "text": {"ebt": "#2a78d6", "transformer": "#eb6834"},
    "mnist": {"ebt": "#2a78d6", "denoiser": "#eb6834"},
    "regression": {
        "regressor": "#2a78d6",
        "ebt_mse": "#eb6834",
        "ebt_contrastive": "#1baf7a",
        "ebt_combined": "#eda100",
        "ebt_regularized": "#e87ba4",
        "mdn": "#008300",
        "ibc": "#6250d6",
        "ired": "#e34948",
    },
}
"""Colour of each model, by task and model name."""
