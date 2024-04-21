import os
import time
from dataclasses import asdict, dataclass, field
from types import SimpleNamespace
from typing import List, Optional
from tqdm import tqdm

import einops
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax
import tyro
from clu import metrics
from flax import jax_utils
from flax.training import common_utils, orbax_utils
from flax.training.train_state import TrainState
from optax import ScaleByAdamState, update_moment, update_moment_per_elem_norm
from optax._src import base, combine, numerics, utils
from optax._src.alias import _scale_by_learning_rate
from rich.console import Console
from rich.pretty import pprint
from torch.utils.data import DataLoader, IterableDataset
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer, FlaxAutoModelForCausalLM, GenerationConfig
from data import DATASET

import orbax.checkpoint
import orbax.checkpoint as ocp

def scale_by_adam_tf_style(
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    eps_root: float = 0.0,
    mu_dtype=None,
) -> base.GradientTransformation:
    """Rescale updates according to the Adam algorithm.

    References:
      [Kingma et al, 2014](https://arxiv.org/abs/1412.6980)

    Args:
      b1: Decay rate for the exponentially weighted average of grads.
      b2: Decay rate for the exponentially weighted average of squared grads.
      eps: Term added to the denominator to improve numerical stability.
      eps_root: Term added to the denominator inside the square-root to improve
        numerical stability when backpropagating gradients through the rescaling.
      mu_dtype: Optional `dtype` to be used for the first order accumulator; if
        `None` then the `dtype` is inferred from `params` and `updates`.

    Returns:
      A `GradientTransformation` object.
    """

    mu_dtype = utils.canonicalize_dtype(mu_dtype)

    def init_fn(params):
        mu = jax.tree_util.tree_map(lambda t: jnp.zeros_like(t, dtype=mu_dtype), params)  # First moment
        nu = jax.tree_util.tree_map(jnp.zeros_like, params)  # Second moment
        return ScaleByAdamState(count=jnp.zeros([], jnp.int32), mu=mu, nu=nu)

    def update_fn(updates, state, params=None):
        del params
        mu = update_moment(updates, state.mu, b1, 1)
        nu = update_moment_per_elem_norm(updates, state.nu, b2, 2)
        count_inc = numerics.safe_int32_increment(state.count)

        ### `optax` default adam implementation
        # mu_hat = bias_correction(mu, b1, count_inc)
        # nu_hat = bias_correction(nu, b2, count_inc)
        # updates = jax.tree_util.tree_map(
        #     lambda m, v: m / (jnp.sqrt(v + eps_root) + eps), mu_hat, nu_hat)
        ### Tensorflow adam implementation
        updates = jax.tree_util.tree_map(
            lambda m, v: (jnp.sqrt(1 - b2**count_inc) / (1 - b1**count_inc)) * m / (jnp.sqrt(v + eps_root) + eps),
            mu,
            nu,
        )  #
        mu = utils.cast_tree(mu, mu_dtype)
        return updates, ScaleByAdamState(count=count_inc, mu=mu, nu=nu)

    return base.GradientTransformation(init_fn, update_fn)


def adam_tf_style(
    learning_rate,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    eps_root: float = 0.0,
    mu_dtype=None,
):
    return combine.chain(
        scale_by_adam_tf_style(b1=b1, b2=b2, eps=eps, eps_root=eps_root, mu_dtype=mu_dtype),
        _scale_by_learning_rate(learning_rate),
    )

