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

def model_policy_forward(
        model,
        input_ids: jnp.ndarray,
    ):
        """Get reward for input_ids."""
        assert input_ids.ndim == 2
        # shape: [batch_size, length]

        # mask out padding tokens
        attention_mask = input_ids != model.generation_config.pad_token_id
        input_ids = jnp.where(attention_mask, input_ids, 0)

        # assign position ids
        position_ids = attention_mask.cumsum(1) - attention_mask

        model_out = model.module.apply(
            variables={"params": model.params},
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids
        )

        # shape: [batch_size, length, 1]
        return model_out

def prepare_policy_forward_and_policy_generate(args, tokenizer):
    """Prepare the forward pass of the policy model and parameters."""

    lm_backbone = FlaxAutoModelForCausalLM.from_pretrained(args.base_model)
    # disable `pad_token_id` and `eos_token_id` because we just want to
    # generate tokens without truncation / padding
    lm_backbone.generation_config.eos_token_id = None
    lm_backbone.generation_config.pad_token_id = tokenizer.pad_token_id

    generation_config = GenerationConfig(
        max_new_tokens=args.task.response_length,
        temperature=args.task.temperature,
        top_k=0.0,
        top_p=1.0,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
    )

    def policy_forward(
        params: LMBackboneParams,
        input_ids: jnp.ndarray,
    ):
        """Get reward for input_ids."""
        assert input_ids.ndim == 2
        # shape: [batch_size, length]

        # mask out padding tokens
        attention_mask = input_ids != tokenizer.pad_token_id
        input_ids = jnp.where(attention_mask, input_ids, 0)

        # assign position ids
        position_ids = attention_mask.cumsum(1) - attention_mask

        lm_backbone_out = lm_backbone.module.apply(
            variables=params.lm_backbone_params,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids
        )

        # shape: [batch_size, length, 1]
        return lm_backbone_out

    def policy_generate(
        params: LMBackboneParams,
        queries: jnp.ndarray,
    ):
        input_ids = queries
        attention_mask = input_ids != tokenizer.pad_token_id
        input_ids = jnp.where(attention_mask, queries, 0)
        output = lm_backbone.generate(
            params=params["params"],
            input_ids=input_ids,
            generation_config=generation_config,
            attention_mask=attention_mask.astype("i4"),
            return_dict_in_generate=True,
        )
        query_length = input_ids.shape[1]
        return jnp.concatenate((queries, output.sequences[:, query_length:]), axis=1)

    key = jax.random.PRNGKey(args.seed)
    key, init_key = jax.random.split(key, 2)
    policy_params = LMBackboneParams(
        lm_backbone_params=flax.core.FrozenDict({"params": lm_backbone.params})
    )

    return policy_forward, policy_generate, policy_params

def model_policy_forward(
        model,
        input_ids: jnp.ndarray,
    ):
        """Get reward for input_ids."""
        assert input_ids.ndim == 2
        # shape: [batch_size, length]

        # mask out padding tokens
        attention_mask = input_ids != model.generation_config.pad_token_id
        input_ids = jnp.where(attention_mask, input_ids, 0)

        # assign position ids
        position_ids = attention_mask.cumsum(1) - attention_mask

        model_out = model.module.apply(
            variables={"params": model.params},
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids
        )

        # shape: [batch_size, length, 1]
        return model_out

def prepare_policy_forward_and_policy_generate(args, tokenizer):
    """Prepare the forward pass of the policy model and parameters."""

    lm_backbone = FlaxAutoModelForCausalLM.from_pretrained(args.base_model)
    # disable `pad_token_id` and `eos_token_id` because we just want to
    # generate tokens without truncation / padding
    lm_backbone.generation_config.eos_token_id = None
    lm_backbone.generation_config.pad_token_id = tokenizer.pad_token_id

    generation_config = GenerationConfig(
        max_new_tokens=args.task.response_length,
        temperature=args.task.temperature,
        top_k=0.0,
        top_p=1.0,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
    )

    def policy_forward(
        params: LMBackboneParams,
        input_ids: jnp.ndarray,
    ):
        """Get reward for input_ids."""
        assert input_ids.ndim == 2
        # shape: [batch_size, length]

        # mask out padding tokens
        attention_mask = input_ids != tokenizer.pad_token_id
        input_ids = jnp.where(attention_mask, input_ids, 0)

        # assign position ids
        position_ids = attention_mask.cumsum(1) - attention_mask

        lm_backbone_out = lm_backbone.module.apply(
            variables=params.lm_backbone_params,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids
        )

        # shape: [batch_size, length, 1]
        return lm_backbone_out

    def policy_generate(
        params: LMBackboneParams,
        queries: jnp.ndarray,
    ):
        input_ids = queries
        attention_mask = input_ids != tokenizer.pad_token_id
        input_ids = jnp.where(attention_mask, queries, 0)
        output = lm_backbone.generate(
            params=params["params"],
            input_ids=input_ids,
            generation_config=generation_config,
            attention_mask=attention_mask.astype("i4"),
            return_dict_in_generate=True,
        )
        query_length = input_ids.shape[1]
        return jnp.concatenate((queries, output.sequences[:, query_length:]), axis=1)

    key = jax.random.PRNGKey(args.seed)
    key, init_key = jax.random.split(key, 2)
    policy_params = LMBackboneParams(
        lm_backbone_params=flax.core.FrozenDict({"params": lm_backbone.params})
    )

    return policy_forward, policy_generate, policy_params