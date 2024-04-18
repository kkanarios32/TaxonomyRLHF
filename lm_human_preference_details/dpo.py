import functools
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

@dataclass
class DPOParams:
    # Batch Size stuff
    local_batch_size: int = 64
    local_mini_batch_size: tyro.conf.Suppress[int] = None
    batch_size: tyro.conf.Suppress[int] = None
    mini_batch_size: tyro.conf.Suppress[int] = None
    gradient_accumulation_steps: int = 16
    """gradient accumulation steps"""
    local_micro_batch_size: tyro.conf.Suppress[int] = None
    """per rank micro batch size"""
    world_size: tyro.conf.Suppress[int] = None
    batch_size: tyro.conf.Suppress[int] = None
    minibatch_size: tyro.conf.Suppress[int] = None
    num_updates: tyro.conf.Suppress[int] = None
    nminibatches: int = 1
    
    # Learning rate, epochs, episodes
    opt_choice = "rmsprop"
    
    total_episodes: int = 11800
    noptepochs: int = 1
    lr: float = 1e-6
    eps: float = 1e-6

    # Learning rate warm-up stuff
    num_warmup: int = 150
    percent_warmup: int = 0.1 # not used

    # DPO params
    beta: float = 0.5 # as suggested in the DPO paper
   

@dataclass
class TaskParams:
    # Query params
    query_length: int = 600 # Changed for DPO
    query_dataset: str = "tldr-dpo" # Changed for DPO
    query_prefix: str = ""
    query_suffix: str = ""
    start_text: Optional[str] = None
    end_text: Optional[str] = None

    # Response params
    response_length: int = 424 # Changed for DPO

    # Truncate response after the first occurrence of this token at or after index after when sampling.
    truncate_token: int = 13
    truncate_after: int = 16
    penalty_reward_value: int = -1

    # LM params
    temperature: float = 0.7



@dataclass
class Args:
    # common args
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    
    seed: int = 1
    """seed of the experiment"""
    
    track: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""
    
    wandb_project_name: str = "dpotrain"
    """the wandb's project name"""
    
    wandb_entity: Optional[str] = None
    """the entity (team) of wandb's project"""
    
    cuda: bool = True
    """Whether to use cuda if available."""
    
    run_name: tyro.conf.Suppress[str] = None
    """TO BE FILLED: a unique name of this run"""

    base_model: str = "kkanarios/gpt2-tldr-sft" # I will probably need to change to the SFT model
    """the name of the pretrained model to use"""
    
    tokenizer_base_model = "gpt2"
    
    print_sample_output_freq: int = 0
    """How often to print sample output"""
    
    save_path: str = "dpo_models/"
    """Where to save the model"""

    task: TaskParams = field(default_factory=TaskParams)
    dpo: DPOParams = field(default_factory=DPOParams)

    # distributed settings
    local_rank: int = 0
    """the rank of this process"""
    
    learner_device_ids: List[int] = field(default_factory=lambda: [0])
    "the device ids that script will use"
    
    learner_devices: tyro.conf.Suppress[int] = None  # real type is `List[str]`
    """the devices that script will use"""
    
    global_learner_devices: tyro.conf.Suppress[int] = None  # real type is `List[str]`
    """the total devices (across all nodes and machines) that script will use"""

    eval_every: int = 50
    save_every: int = 300

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

# a pytorch dataset
class MyDPODataset(IterableDataset):
    def __init__(self, generator, tokenizer, query_length, seed, start_text=None, end_text=None, split="train"):
        self.generator = generator
        self.tokenizer = tokenizer
        self.query_length = query_length
        self.start_text = start_text
        self.end_text = end_text
        self.seed = seed
        token_to_index = tokenizer.get_vocab()
        self.start_token = token_to_index[start_text] if self.start_text else None
        self.end_token = token_to_index[end_text] if self.end_text else None
        self.split = split

    def __iter__(self):
        for query, response_pref, response_rej in self.generator(self.split, self.seed, shuffle=False):
            query_tokens = self.tokenizer.encode(query)

            if self.start_token is not None:
                try:
                    first_index = query_tokens.index(self.start_token) + 1
                    if first_index < len(query_tokens):
                        query_tokens = query_tokens[first_index:]
                except:
                    continue

            query_tokens = query_tokens[: self.query_length]
            if self.end_token is not None:
                try:
                    last_index = len(query_tokens) - query_tokens[::-1].index(self.end_token)
                    query_tokens = query_tokens[:last_index]
                except:
                    continue

            query_output = self.tokenizer.pad(
                {"input_ids": query_tokens},
                padding="max_length",
                max_length=self.query_length,
                return_tensors="np",
                return_attention_mask=False,
            )

            max_response_length = self.tokenizer.model_max_length - self.query_length

            # For preferred responses
            response_tokens_pref = self.tokenizer.encode(response_pref, max_length=max_response_length,
                                                 truncation=True)
            response_output_pref = self.tokenizer.pad({"input_ids": response_tokens_pref},
                                                 padding="max_length",
                                                 max_length=max_response_length,
                                                 return_tensors = "np",
                                                 return_attention_mask=False
                                                )
            
            # For rejected responses
            response_tokens_rej = self.tokenizer.encode(response_rej, max_length=max_response_length,
                                                 truncation=True)
            response_output_rej = self.tokenizer.pad({"input_ids": response_tokens_rej},
                                                 padding="max_length",
                                                 max_length=max_response_length,
                                                 return_tensors = "np",
                                                 return_attention_mask=False
                                                )

            yield query_output["input_ids"], np.squeeze(response_output_pref["input_ids"]), np.squeeze(response_output_rej["input_ids"])


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
        batch_size=args.dpo.batch_size,
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


def ceil_div(a, b):
    return (a - 1) // b + 1


def exact_div(a, b):
    q = a // b
    if a != q * b:
        raise ValueError(f"Inexact division: {a} / {b} = {a / b}")
    return q


@flax.struct.dataclass
class LMBackboneParams:
    """Parameters for the language model backbone."""

    lm_backbone_params: flax.core.FrozenDict
    
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

@flax.struct.dataclass
class DPOStatistics(metrics.Collection):
    loss: metrics.Average.from_output("loss")


def compute_loss(params, 
                 apply_fn,
                 ref_model, # reference model
                 mb_query_responses_pref,
                 mb_query_responses_rej,
                 tokenizer,
                 args  
                ):

        """
        Implementing preference loss based on https://github.com/eric-mitchell/direct-preference-optimization/blob/main/trainers.py#L45
        """

        # From policy
        output_pref_theta = apply_fn(params, mb_query_responses_pref)
        output_rej_theta = apply_fn(params, mb_query_responses_rej)

        logits_pref_theta =  output_pref_theta.logits[:, args.task.query_length - 1 : -1, :] / args.task.temperature
        logits_rej_theta =  output_rej_theta.logits[:, args.task.query_length - 1 : -1, :] / args.task.temperature

        # From reference model
        output_pref_refm = model_policy_forward(ref_model, mb_query_responses_pref)
        output_rej_refm = model_policy_forward(ref_model, mb_query_responses_rej)
        
        logits_pref_refm =  output_pref_refm.logits[:, args.task.query_length - 1 : -1, :] / args.task.temperature
        logits_rej_refm =  output_rej_refm.logits[:, args.task.query_length - 1 : -1, :] / args.task.temperature

        # Processing responses
        responses_pref = mb_query_responses_pref[:, args.task.query_length :] 
        responses_rej = mb_query_responses_rej[:, args.task.query_length :] 

        # Using the same variable names from the DPO implmentation linked above
        policy_chosen_logps = -optax.softmax_cross_entropy_with_integer_labels(logits_pref_theta, responses_pref)
        policy_rejected_logps = -optax.softmax_cross_entropy_with_integer_labels(logits_rej_theta, responses_rej)

        reference_chosen_logps = -optax.softmax_cross_entropy_with_integer_labels(logits_pref_refm, responses_pref)
        reference_rejected_logps = -optax.softmax_cross_entropy_with_integer_labels(logits_rej_refm, responses_rej)        

        # Some filtering thing for padded tokens
        filter_for_pad_logprobs_pref = (responses_pref!=tokenizer.pad_token_id) # TODO: Do I need both of these or should they be the same?
        filter_for_pad_logprobs_rej = (responses_rej!=tokenizer.pad_token_id)

        policy_chosen_logps = policy_chosen_logps*filter_for_pad_logprobs_pref
        policy_rejected_logps = policy_rejected_logps*filter_for_pad_logprobs_rej

        reference_chosen_logps = reference_chosen_logps*filter_for_pad_logprobs_pref
        reference_rejected_logps = reference_rejected_logps*filter_for_pad_logprobs_rej

        # Computing loss
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps

        temp = pi_logratios - ref_logratios
        temp = jnp.sum(temp, axis=1)
        assert temp.ndim == 1
        # dpo_loss = jnp.sum(temp)
        
        dpo_loss = -flax.linen.log_sigmoid(args.dpo.beta*temp)
        
        dpo_loss_val = jnp.sum(dpo_loss)

        # dpo_loss_val = jnp.sum(policy_chosen_logps) # this should boil down to SFT if you want to debug
        
        current_dpo_stats = dict(loss=dpo_loss_val)

        return dpo_loss_val, current_dpo_stats


def train_step(
    policy_state, # with respect to the current model
    ref_model, # with respect to the reference model
    dpo_stats,
    mb_query_responses_pref,
    mb_query_responses_rej,
    tokenizer,
    args
):

    loss_fn = functools.partial(
        compute_loss,
        apply_fn=policy_state.apply_fn,
        ref_model=ref_model,
        mb_query_responses_pref=mb_query_responses_pref,
        mb_query_responses_rej=mb_query_responses_rej,        
        tokenizer=tokenizer,
        args=args
    )

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, current_dpo_stats), grads = grad_fn(policy_state.params)
    grads = jax.lax.pmean(grads, "batch")
    policy_state = policy_state.apply_gradients(grads=grads)
    
    dpo_stats = dpo_stats.merge(DPOStatistics.gather_from_model_output(**current_dpo_stats))

    return policy_state, dpo_stats


def eval_step(
        policy_state,
        ref_model,
        mb_query_responses_pref,
        mb_query_responses_rej,        
        tokenizer,
        args,        
):
    loss_fn = functools.partial(
        compute_loss,
        apply_fn=policy_state.apply_fn,
        ref_model=ref_model,
        mb_query_responses_pref=mb_query_responses_pref,
        mb_query_responses_rej=mb_query_responses_rej,        
        tokenizer=tokenizer,
        args=args
    )

    params = jax.lax.stop_gradient(policy_state.params)
    loss, _ = loss_fn(params)

    return loss


def linear_schedule(count, args):
    # anneal learning rate linearly
    frac = 1.0 - (count // (args.dpo.nminibatches * args.dpo.noptepochs)) / args.dpo.num_updates
    return args.dpo.lr * frac

def linear_warmup_schedule(count, args):
    frac = jnp.min(jnp.array([1.0, (count // (args.dpo.nminibatches * args.dpo.noptepochs)) / (args.dpo.num_warmup)]))
    return args.dpo.lr * frac

def cosine_schedule(count, args):
    return optax.cosine_decay_schedule(init_value=args.dpo.lr, decay_steps = args.dpo.num_updates, alpha=0)(count)

def train(args: Args):
    local_devices = jax.local_devices()
    global_devices = jax.devices()
    args.dpo.world_size = jax.process_count()
    learner_devices = [local_devices[d_id] for d_id in args.learner_device_ids]
    global_learner_devices = [
        global_devices[d_id + process_index * len(local_devices)]
        for process_index in range(args.dpo.world_size)
        for d_id in args.learner_device_ids
    ]
    pprint({"global_learner_devices": global_learner_devices})
    args.global_learner_devices = [str(item) for item in global_learner_devices]
    args.learner_devices = [str(item) for item in learner_devices]
    args.local_rank = jax.process_index()
    args.dpo.batch_size = int(args.dpo.local_batch_size * len(args.learner_devices) * args.dpo.world_size)
    args.dpo.minibatch_size = exact_div(args.dpo.batch_size, args.dpo.nminibatches)
    args.dpo.local_mini_batch_size = exact_div(args.dpo.local_batch_size, args.dpo.nminibatches)
    args.dpo.local_micro_batch_size = 1 # exact_div(args.dpo.local_mini_batch_size, args.dpo.gradient_accumulation_steps)
    # `per_rank_rollout_batch_size` is our `args.dpo.local_batch_size`
    # `per_rank_minibatch_size` is our `args.dpo.local_mini_batch_size`
    args.dpo.num_updates = args.dpo.total_episodes // args.dpo.batch_size

    console = Console(force_terminal=True)
    run_name = f"{args.exp_name}__{args.seed}__{int(time.time())}"
    writer = SimpleNamespace()  # dummy writer
    writer.add_scalar = lambda x, y, z: None
    writer.add_histogram = lambda x, y, z: None

    if args.local_rank == 0:
        if args.track:
            import wandb

            wandb.init(
                project=args.wandb_project_name,
                entity=args.wandb_entity,
                sync_tensorboard=True,
                config=asdict(args),
                name=run_name,
                save_code=True,
            )
            wandb.run.log_code(".")
        writer = SummaryWriter(f"runs/{run_name}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )
        pprint(args)
    local_seed = args.seed + args.local_rank * 100003  # Prime
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_base_model,
        padding_side="right",
    )
    # we use the padding token manually but do not resize the token embedding of the model
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    
    print("tokenizer initialized")
    
    (
        policy_forward,
        policy_generate,
        policy_params,
    ) = prepare_policy_forward_and_policy_generate(args, tokenizer)
    
    ref_model = FlaxAutoModelForCausalLM.from_pretrained(args.base_model)
    # disable `pad_token_id` and `eos_token_id` because we just want to
    # generate tokens without truncation / padding
    ref_model.generation_config.eos_token_id = None
    ref_model.generation_config.pad_token_id = tokenizer.pad_token_id

    generation_config = GenerationConfig(
        max_new_tokens=args.task.response_length,
        temperature=args.task.temperature,
        top_k=0.0,
        top_p=1.0,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
    )
    
    print("policy param initialized")

    if (args.dpo.opt_choice=="adam_tf"):
        optim_choice = adam_tf_style
    elif (args.dpo.opt_choice=="rmsprop"):
        optim_choice = optax.rmsprop
    elif (args.dpo.opt_choice=="adamw"):
        optim_choice = optax.adamw
    else:
        optim_choice = optax.adam
    

    optimizer = optax.MultiSteps(
        optax.inject_hyperparams(optim_choice)(
            learning_rate = functools.partial(linear_warmup_schedule, args=args), 
            eps=args.dpo.eps,
        ),
        every_k_schedule=args.dpo.gradient_accumulation_steps,
    )
    
    print("optimizer initialized")

    policy_state = TrainState.create(apply_fn=policy_forward, params=policy_params, tx=optimizer)
    policy_state = jax_utils.replicate(policy_state)
    
    print("Train state created")

    train_dataloader = get_batch_loader(tokenizer, args, seed=local_seed, split='train')
    eval_dataloader = get_batch_loader(tokenizer, args, seed=local_seed, split='test')

    # Changed to DPO
    dataset = MyDPODataset(
        DATASET[args.task.query_dataset],
        tokenizer,
        args.task.query_length,
        seed=local_seed,
        start_text=args.task.start_text,
        end_text=args.task.end_text,
    )
    
    print("dataset initialized")
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.dpo.batch_size,
        collate_fn=numpy_collate,
    )
    iter_dataloader = iter(dataloader)
    
    print("Iterable dataloader initialized")

    # Changed to have multiple responses
    def train_update(policy_state, input_ids, response_pref_ids, response_rej_ids, dpo_stats):
        queries = right_padding_to_left_padding(input_ids, tokenizer.pad_token_id)

        query_responses_pref = jnp.concatenate((input_ids, response_pref_ids), axis=1)
        query_responses_rej = jnp.concatenate((input_ids, response_rej_ids), axis=1)

        def dpo_single_microbatch(carry, inp):
            policy_state, dpo_stats = carry
            mb_query_responses_pref, mb_query_responses_rej = inp

            policy_state, dpo_stats = train_step(
                policy_state=policy_state,
                ref_model=ref_model, # added for reference policy
                dpo_stats=dpo_stats,
                mb_query_responses_pref=mb_query_responses_pref,
                mb_query_responses_rej=mb_query_responses_rej,
                tokenizer=tokenizer,
                args=args
            )
            return (policy_state, dpo_stats), None

        def dpo_single_epoch(carry, inp):
            policy_state, dpo_stats, key = carry
            key, subkey = jax.random.split(key, 2)
            perm = jax.random.permutation(key, args.dpo.local_batch_size)
            # That is -> query_responses, logprobs = inp
            
            # For both rejected and preferred query responses
            mbs_query_responses_pref = einops.rearrange(
                query_responses_pref[perm],
                "(c m) l -> c m l",
                c=args.dpo.gradient_accumulation_steps,
            )

            mbs_query_responses_rej = einops.rearrange(
                query_responses_rej[perm],
                "(c m) l -> c m l",
                c=args.dpo.gradient_accumulation_steps,
            )

            (policy_state, dpo_stats), _ = jax.lax.scan(
                f=dpo_single_microbatch,
                init=(policy_state, dpo_stats),
                xs=(
                    mbs_query_responses_pref,
                    mbs_query_responses_rej,
                ),
            )
            return (policy_state, dpo_stats, key), None

        key = jax.random.PRNGKey(args.seed)
        # Do multiple epochs of DPO training, with a fresh random shuffle in each epoch
        (policy_state, dpo_stats, _), _ = jax.lax.scan(
            f=dpo_single_epoch,
            init=(policy_state, dpo_stats, key),
            xs=None,
            length=args.dpo.noptepochs,
        )

        dpo_stats = jax.lax.pmean(dpo_stats.compute(), "batch")

        samples_to_print = dict(
            query_response_pref=query_responses_pref[0] # Printing preferred responses
        )
        return policy_state, dpo_stats, samples_to_print

    p_train_update = jax.pmap(
        train_update,
        axis_name="batch",
        donate_argnums=(0,),
    )
    
    def eval(policy_state, input_ids, response_pref_ids, response_rej_ids):
        queries = right_padding_to_left_padding(input_ids, tokenizer.pad_token_id)

        query_responses_pref = jnp.concatenate((input_ids, response_pref_ids), axis=1)
        query_responses_rej = jnp.concatenate((input_ids, response_rej_ids), axis=1)

        def dpo_single_microbatch(inp):
            mb_query_responses_pref, mb_query_responses_rej = inp

            loss_val = eval_step(
                policy_state=policy_state,
                mb_query_responses_pref=mb_query_responses_pref,
                mb_query_responses_rej=mb_query_responses_rej,
                tokenizer=tokenizer,
                args=args,
            )
            return loss_val
        
        # That is -> query_responses, logprobs = inp
            
        # For both rejected and preferred query responses
        mbs_query_responses_pref = einops.rearrange(
                query_responses_pref,
                "(c m) l -> c m l",
                c=args.dpo.gradient_accumulation_steps,
        )

        mbs_query_responses_rej = einops.rearrange(
                query_responses_rej,
                "(c m) l -> c m l",
                c=args.dpo.gradient_accumulation_steps,
        )

        eval_epoch = jax.vmap(dpo_single_microbatch, in_axes=0)
        dpo_stats = jnp.sum(eval_epoch(mbs_query_responses_pref, mbs_query_responses_rej))
        dpo_stats = jax.lax.pmean(dpo_stats, "batch")

        samples_to_print = dict(
            query_response_pref=query_responses_pref[0] # Printing preferred responses
        )

        return dpo_stats, samples_to_print

    p_eval = jax.pmap(
        eval,
        axis_name="batch",
        donate_argnums=(0,),
    )

    print("initialized eval")

    global_step = 0
    
    print("starting train loop")

    # Changed this to have both responses
    for update, [input_ids, response_pref_ids, response_rej_ids] in tqdm(enumerate(train_dataloader)):
        # doing eval stuff
        if (args.local_rank == 0) and (update%args.eval_every==0) and (update>0):
            losses = []

            for eval_batch, [eval_input_ids, eval_response_pref_ids, eval_response_rej_ids] in tqdm(enumerate(eval_dataloader)):
                eval_input_ids = common_utils.shard(eval_input_ids)

                # For both responses
                eval_response_pref_ids = common_utils.shard(eval_response_pref_ids)
                eval_response_rej_ids = common_utils.shard(eval_response_rej_ids)

                dpo_stats, samples_to_print = p_eval(
                    policy_state=policy_state,
                    input_ids=eval_input_ids,
                    response_pref_ids=eval_response_pref_ids, 
                    response_rej_ids=eval_response_pref_ids,
                )
                losses.append(dpo_stats)

            loss = np.mean(np.array(losses))

            writer.add_scalar("dpo/eval_loss/total", loss, update)

    
        global_step += args.dpo.batch_size
        input_ids = common_utils.shard(input_ids)

        # For both responses
        response_pref_ids = common_utils.shard(response_pref_ids)
        response_rej_ids = common_utils.shard(response_rej_ids)

        dpo_stats = jax_utils.replicate(DPOStatistics.empty())
        policy_state, dpo_stats, samples_to_print = p_train_update(
            policy_state=policy_state,
            input_ids=input_ids,
            response_pref_ids = response_pref_ids,
            response_rej_ids = response_rej_ids,
            dpo_stats=dpo_stats
        )
        
        if (args.local_rank == 0) and (update%args.save_every==0): # (update>0) and (update%1000==0):
            if args.save_path:
                ckpt = {"policy_model": jax_utils.unreplicate(policy_state), "args": vars(args)}
                orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
                save_args = orbax_utils.save_args_from_target(ckpt)
                orbax_checkpointer.save(args.save_path + f"model_{update}", ckpt, save_args=save_args, force=True)

#             if args.local_rank == 0 and args.track:
#                 wandb.finish()

#         try:
#             sample_query_response = samples_to_print["query_response_pref"][0]
#             console.print(
#                 f"[green][bold]{'Query'}:[/]\n"
#                 + f"[green]{ tokenizer.decode(sample_query_response[:args.task.query_length], skip_special_tokens=True)}[/]\n\n"
#                 + f"[yellow][bold]{'Response'}:[/]\n"
#                 )
#         except Exception as e:
#             print(e)

        
        # RL metrics aggregated at the batch level
        dpo_stats = common_utils.get_metrics([dpo_stats])
        writer.add_scalar("dpo/loss/total", dpo_stats["loss"].item(), update)

        # Logging learning rate and learning progress
        writer.add_scalar("dpo/lr", policy_state.opt_state.inner_opt_state.hyperparams["learning_rate"][0].item(), update)
        writer.add_scalar("dpo/episode", global_step, update)

    print("finished training")
    
    policy_state = jax_utils.unreplicate(policy_state)
    # save model
    if args.local_rank == 0:
        if args.save_path:
            ckpt = {"policy_model": policy_state, "args": vars(args)}
            orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
            save_args = orbax_utils.save_args_from_target(ckpt)
            orbax_checkpointer.save(args.save_path, ckpt, save_args=save_args, force=True)

        if args.local_rank == 0 and args.track:
            wandb.finish()


if __name__ == "__main__":
    args = tyro.cli(Args)
    print("args initialized")
    train(args)
