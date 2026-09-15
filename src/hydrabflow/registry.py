"""Everything you can register, in one file.

Five extension points share one mechanism: a name -> object dict filled by decorators.
Drop a module into the matching package, decorate what it defines, and select it by name in config.

    from hydrabflow.registry import register_simulator

    @register_simulator("my_model")
    class MySimulator(BaseSimulator): ...

``discover()`` (called by each package's ``__init__``) imports every module in the package, which is
what makes dropping a file sufficient — no ``__init__.py`` edit.

| decorator                     | package                    | selected by                    |
|-------------------------------|----------------------------|--------------------------------|
| ``@register_simulator``       | ``simulators/``            | ``simulator=<name>``           |
| ``@register_step``            | ``preprocessing/``         | ``preprocessing.steps[].name`` |
| ``@register_augmentation``    | ``augmentation/``          | ``augmentation.steps[]``       |
| ``@register_summary_network`` | ``networks/``              | ``model.summary_network.type`` |
| ``@register_inference_network``| ``networks/``             | ``model.inference_network.type``|
"""

from __future__ import annotations

import importlib
import pkgutil
from typing import Any, Callable, Dict, Generic, Iterable, List, TypeVar

import numpy as np

T = TypeVar("T")


class Registry(Generic[T]):
    """A named collection filled by ``@registry.add("name")`` decorators.

    ``package`` is where its components live. The first lookup imports every module in it (see
    :func:`discover`), which is what makes dropping a file sufficient — no ``__init__.py`` edit.
    Discovery is lazy so a config-only context never imports the compute backend.
    """

    def __init__(self, kind: str, package: str) -> None:
        self.kind = kind
        self.package = package
        self.items: Dict[str, T] = {}
        self._discovered = False

    def add(self, name: str) -> Callable[[T], T]:
        def _wrap(obj: T) -> T:
            if self.items.setdefault(name, obj) is not obj:
                raise ValueError(f"{self.kind} '{name}' is already registered")
            return obj

        return _wrap

    def discover(self) -> Registry[T]:
        """Import ``self.package``'s modules so their decorators have run. Idempotent."""
        if not self._discovered:
            self._discovered = True  # set first: the imports below re-enter this module
            module = importlib.import_module(self.package)
            discover(self.package, module.__path__)
        return self

    def get(self, name: str) -> T:
        if name not in self.items:
            self.discover()
        if name not in self.items:
            raise KeyError(
                f"Unknown {self.kind} '{name}'. Registered: {sorted(self.items)}. "
                f"To add one, drop a module in {self.package.replace('.', '/')}/ and decorate it."
            )
        return self.items[name]


def discover(package_name: str, package_path: Iterable[str]) -> None:
    """Import every non-underscore module in a package, so its decorators run."""
    for info in pkgutil.iter_modules(list(package_path)):
        if not info.name.startswith("_"):
            importlib.import_module(f"{package_name}.{info.name}")


SIMULATORS: Registry[type] = Registry("simulator", "hydrabflow.simulators")
STEPS: Registry[type] = Registry("preprocessing step", "hydrabflow.preprocessing")
AUGMENTATIONS: Registry[Callable[..., Callable[[dict], dict]]] = Registry(
    "augmentation", "hydrabflow.augmentation"
)
SUMMARY_NETWORKS: Registry[Callable[[Any], Any]] = Registry(
    "summary_network.type", "hydrabflow.networks"
)
INFERENCE_NETWORKS: Registry[Callable[[Any], Any]] = Registry(
    "inference_network.type", "hydrabflow.networks"
)

register_simulator = SIMULATORS.add
register_step = STEPS.add
register_augmentation = AUGMENTATIONS.add
register_summary_network = SUMMARY_NETWORKS.add
register_inference_network = INFERENCE_NETWORKS.add

Augmentation = Callable[[dict], dict]


# Builders. Imports are local so this module never imports the packages that import it.


def get_simulator(cfg) -> Any:
    """Instantiate the simulator selected by ``cfg.simulator``."""
    return SIMULATORS.get(cfg.name)(params=cfg.params)


def build_pipeline(cfg) -> Any:
    """Build the preprocessing pipeline from ``cfg.preprocessing``.

    Each entry in ``cfg.steps`` is ``{name: <registry key>, ...params}``; the rest are kwargs.
    """
    from omegaconf import OmegaConf

    from hydrabflow.preprocessing.base import PreprocessPipeline

    steps = []
    for entry in OmegaConf.to_container(cfg.steps, resolve=True):
        entry = dict(entry)
        steps.append(STEPS.get(entry.pop("name"))(**entry))
    return PreprocessPipeline(steps)


def build_augmentations(
    cfg, rng: np.random.Generator | None = None, context: dict | None = None
) -> List[Augmentation]:
    """Build the ordered augmentation list from ``cfg.augmentation``.

    Each step gets its own child generator (``rng.spawn``), so its random stream does not depend on
    which other steps are enabled or on their order.
    """
    from omegaconf import OmegaConf

    params = OmegaConf.to_container(cfg.params, resolve=True)
    steps = list(cfg.steps)
    rng = rng if rng is not None else np.random.default_rng()
    return [
        AUGMENTATIONS.get(name)(params, child, context or {})
        for name, child in zip(steps, rng.spawn(len(steps)))
    ]


def build_summary_network(cfg, summary_variables: Iterable[str] | None = None) -> Any:
    """Return the summary network selected by ``cfg.type``.

    One key (the default): that one network. Several keys: the adapter groups them, so one backbone
    is built per key and combined by ``bf.networks.FusionNetwork``. Give a key its own architecture
    with ``model.summary_network.params.backbones={<key>: <type>}``; the rest use ``cfg.type``.
    A builder flagged ``consumes_grouped_inputs = True`` (the stream ``fusion`` network) receives
    the whole config instead and does its own per-key construction.
    """
    keys = list(summary_variables or [])
    builder = SUMMARY_NETWORKS.get(cfg.type)
    # A builder that already consumes the grouped dict itself (e.g. ``fusion``, which builds one
    # backbone per key from ``params.backbones`` and routes the attention mask) is used as-is.
    if len(keys) < 2 or getattr(builder, "consumes_grouped_inputs", False):
        return builder(cfg)

    import bayesflow as bf
    import keras

    types = dict(cfg.params.get("backbones", {}))
    backbones = {k: SUMMARY_NETWORKS.get(types.get(k, cfg.type))(cfg) for k in keys}
    head = keras.Sequential(
        [
            bf.networks.MLP(widths=[int(cfg.mlp_width)] * int(cfg.mlp_depth)),
            keras.layers.Dense(units=int(cfg.summary_dim)),
        ]
    )
    return bf.networks.FusionNetwork(backbones, head=head)


def build_inference_network(cfg, model_cfg=None) -> Any:
    """Return the inference (posterior) network selected by ``cfg.type``.

    A builder flagged ``needs_model_cfg = True`` (``grouped_diffusion``, which reads the summary
    network's per-backbone widths to know where its condition groups start and end) also receives
    the enclosing ``cfg.model``.
    """
    builder = INFERENCE_NETWORKS.get(cfg.type)
    if getattr(builder, "needs_model_cfg", False):
        return builder(cfg, model_cfg)
    return builder(cfg)
