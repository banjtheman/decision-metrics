"""The same option descriptions for HTTP inference and supervised examples."""
import json

OPTION_RENDERING = "json-descriptions-v1"


def render_options(options: dict) -> dict[str, str]:
    """Keep text unchanged; encode other JSON values without losing fields or order."""
    return {name: value if isinstance(value, str) else json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
        for name, value in options.items()}
