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

def get_batch_loader(tokenizer, args, seed=0, split="train"):
    dataset = MyDPODataset(
        DATASET[args.task.query_dataset],
        tokenizer,
        args.task.query_length,
        seed=seed,
        start_text=args.task.start_text,
        end_text=args.task.end_text,
        split=split,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.train.batch_size,
        collate_fn=numpy_collate,
        drop_last=True
    )

    return dataloader

def get_batch_loader_eval(tokenizer, args, seed=0, split="train"):
    dataset = MyDPODataset(
        DATASET[args.task.query_dataset],
        tokenizer,
        args.task.query_length,
        seed=seed,
        start_text=args.task.start_text,
        end_text=args.task.end_text,
        split=split,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.train.eval_batch_size,
        collate_fn=numpy_collate,
        drop_last=True
    )

    return dataloader

def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)


def right_padding_to_left_padding(tokens, pad_id):
    def pad_row(row):
        mask = 1 - (row == pad_id)  # 1 if not pad_id, 0 if pad_id
        return row[jnp.argsort(mask)]  # uses the fact that jnp.argsort is stable by default

    return jax.vmap(pad_row)(tokens)