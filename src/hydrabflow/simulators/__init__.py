"""Forward models. Drop a module here; ``@register_simulator`` makes it selectable by name."""

from hydrabflow.utils.registry import discover

discover(__name__, __path__)
