"""The same option, level and criteria descriptions for HTTP inference and supervised examples."""
import json

OPTION_RENDERING = "json-descriptions-v1"


def render_value(value):
    """Keep text unchanged; encode other JSON values without losing fields or order."""
    return value if isinstance(value, str) else json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def render_options(options: dict) -> dict[str, str]:
    return {name: render_value(value) for name, value in options.items()}
