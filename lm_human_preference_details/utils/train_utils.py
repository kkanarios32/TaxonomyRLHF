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

def linear_schedule(count, args):
    # anneal learning rate linearly
    frac = 1.0 - (count // (args.sft.nminibatches * args.sft.noptepochs)) / args.sft.num_updates
    return args.sft.lr * frac


def constant_schedule(count, args):
    # anneal learning rate linearly
    return args.sft.lr

def cosine_schedule(count, args):
    return optax.cosine_decay_schedule(init_value=args.sft.lr, decay_steps = args.sft.num_updates, alpha=1e-7)(count)


def linear_warmup_schedule(count, args):
    frac = jnp.min(jnp.array([1.0, (count // (args.sft.nminibatches * args.sft.noptepochs)) / (args.sft.num_warmup)]))
    return args.sft.lr * frac