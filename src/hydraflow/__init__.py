"""HydraFlow: a reusable BayesFlow + Hydra SBI pipeline template.

Importing the package runs :mod:`hydraflow.utils.backend` first, which (a) pins the visible
GPUs via ``autocvd`` and (b) pins the Keras backend to JAX — both *before* any keras/bayesflow/JAX
import can happen, so downstream modules can import keras safely. See that module for the
``HYDRAFLOW_NUM_GPUS`` / ``CUDA_VISIBLE_DEVICES`` / ``KERAS_BACKEND`` overrides.
"""

from hydraflow.utils import backend as _backend  # noqa: F401  (side effect: sets KERAS_BACKEND)

__version__ = "0.1.0"
