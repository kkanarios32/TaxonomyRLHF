import copy
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

@dataclass
class SFTParams:
    #Batch Size stuff
    local_batch_size: int = 8
    local_mini_batch_size: tyro.conf.Suppress[int] = None
    batch_size: tyro.conf.Suppress[int] = None
    mini_batch_size: tyro.conf.Suppress[int] = None
    gradient_accumulation_steps: int = 1
    """gradient accumulation steps"""
    local_micro_batch_size: tyro.conf.Suppress[int] = None
    """per rank micro batch size"""
    world_size: tyro.conf.Suppress[int] = None
    batch_size: tyro.conf.Suppress[int] = None
    minibatch_size: tyro.conf.Suppress[int] = None
    num_updates: tyro.conf.Suppress[int] = None
    nminibatches: int = 1
    
    # Learning rate, epochs, episodes
    total_episodes: int = 1000000
    noptepochs: int = 1
    lr: float = 0.00001
    eps: float = 1e-5


@dataclass
class TaskParams:
    # Query params
    query_length: int = 600
    query_dataset: str = "tldr-sft"
    query_prefix: str = ""
    query_suffix: str = ""
    start_text: Optional[str] = None
    end_text: Optional[str] = None

    # Response params
    response_length: int = 400

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
    
    wandb_project_name: str = "sfttrain"
    """the wandb's project name"""
    
    wandb_entity: Optional[str] = None
    """the entity (team) of wandb's project"""
    
    cuda: bool = True
    """Whether to use cuda if available."""
    
    run_name: tyro.conf.Suppress[str] = None
    """TO BE FILLED: a unique name of this run"""

    base_model: str = "gpt2"
    """the name of the pretrained model to use"""
    
    print_sample_output_freq: int = 0
    """How often to print sample output"""
    
    save_path: str = "sftmodels/"
    """Where to save the model"""
    
    use_tensorflow_adam: bool = True
    """Whether to use tensorflow-style Adam optimizer instead of PyTorch's"""
    
    task: TaskParams = field(default_factory=TaskParams)
    sft: SFTParams = field(default_factory=SFTParams)

    # distributed settings
    local_rank: int = 0
    """the rank of this process"""
    
    learner_device_ids: List[int] = field(default_factory=lambda: [0])
    "the device ids that script will use"
    
    learner_devices: tyro.conf.Suppress[int] = None  # real type is `List[str]`
    """the devices that script will use"""
    
    global_learner_decices: tyro.conf.Suppress[int] = None  # real type is `List[str]`
    """the total devices (across all nodes and machines) that script will use"""


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
class MySFTDataset(IterableDataset):
    def __init__(self, generator, tokenizer, query_length, seed, start_text=None, end_text=None):
        self.generator = generator
        self.tokenizer = tokenizer
        self.query_length = query_length
        self.start_text = start_text
        self.end_text = end_text
        self.seed = seed
        token_to_index = tokenizer.get_vocab()
        self.start_token = token_to_index[start_text] if self.start_text else None
        self.end_token = token_to_index[end_text] if self.end_text else None

    def __iter__(self):
        for query, response in self.generator("train", self.seed, shuffle=False):
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
            response_tokens = self.tokenizer.encode(response, max_length=max_response_length,
                                                 truncation=True)
            response_output = self.tokenizer.pad({"input_ids": response_tokens},
                                                 padding="max_length",
                                                 max_length=max_response_length,
                                                 return_tensors = "np",
                                                 return_attention_mask=False
                                                )

            yield query_output["input_ids"], np.squeeze(response_output["input_ids"])



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
class SFTStatistics(metrics.Collection):
    loss: metrics.Average.from_output("loss")

def train_step(
    policy_state,
    sft_stats,
    mb_query_responses,
    tokenizer,
    args
):
    def sft_loss(params):
        output = policy_state.apply_fn(params, mb_query_responses)

        logits = output.logits[:, args.task.query_length - 1 : -1, :]
        logits /= args.task.temperature
        responses = mb_query_responses[:, args.task.query_length :]
        
        response_logprobs = -optax.softmax_cross_entropy_with_integer_labels(logits, responses)
        filter_for_pad_logprobs = (responses!=tokenizer.pad_token_id)
        response_logprobs=response_logprobs*filter_for_pad_logprobs
        
        sft_loss_val = jnp.sum(response_logprobs)
                
        current_sft_stats = dict(loss=sft_loss_val)

        return sft_loss_val, current_sft_stats

    grad_fn = jax.value_and_grad(sft_loss, has_aux=True)
    (loss, current_sft_stats), grads = grad_fn(policy_state.params)
    grads = jax.lax.pmean(grads, "batch")
    policy_state = policy_state.apply_gradients(grads=grads)
    
    sft_stats = sft_stats.merge(SFTStatistics.gather_from_model_output(**current_sft_stats))

    return policy_state, sft_stats


def linear_schedule(count, args):
    # anneal learning rate linearly
    frac = 1.0 - (count // (args.sft.nminibatches * args.sft.noptepochs)) / args.sft.num_updates
    return args.sft.lr * frac

def constant_schedule(count, args):
    # anneal learning rate linearly
    return args.sft.lr


def train(args: Args):
    local_devices = jax.local_devices()
    global_devices = jax.devices()
    args.sft.world_size = jax.process_count()
    learner_devices = [local_devices[d_id] for d_id in args.learner_device_ids]
    global_learner_decices = [
        global_devices[d_id + process_index * len(local_devices)]
        for process_index in range(args.sft.world_size)
        for d_id in args.learner_device_ids
    ]
    pprint({"global_learner_decices": global_learner_decices})
    args.global_learner_decices = [str(item) for item in global_learner_decices]
    args.learner_devices = [str(item) for item in learner_devices]
    args.local_rank = jax.process_index()
    args.sft.batch_size = int(args.sft.local_batch_size * len(args.learner_devices) * args.sft.world_size)
    args.sft.minibatch_size = exact_div(args.sft.batch_size, args.sft.nminibatches)
    args.sft.local_mini_batch_size = exact_div(args.sft.local_batch_size, args.sft.nminibatches)
    args.sft.local_micro_batch_size = 1 # exact_div(args.sft.local_mini_batch_size, args.sft.gradient_accumulation_steps)
    # `per_rank_rollout_batch_size` is our `args.sft.local_batch_size`
    # `per_rank_minibatch_size` is our `args.sft.local_mini_batch_size`
    args.sft.num_updates = args.sft.total_episodes // args.sft.batch_size

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
        args.base_model,
        padding_side="right",
    )
    # we use the padding token manually but do not resize the token embedding of the model
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        
    (
        policy_forward,
        policy_generate,
        policy_params,
    ) = prepare_policy_forward_and_policy_generate(args, tokenizer)
    ref_policy_params = copy.deepcopy(policy_params)
    
    if args.use_tensorflow_adam:
        adam = adam_tf_style
    else:
        adam = optax.adam

    optimizer = optax.MultiSteps(
        optax.inject_hyperparams(adam)(
            learning_rate=functools.partial(constant_schedule, args=args),
            eps=args.sft.eps,
        ),
        every_k_schedule=args.sft.gradient_accumulation_steps,
    )
    
    policy_state = TrainState.create(apply_fn=policy_forward, params=policy_params, tx=optimizer)
    policy_state = jax_utils.replicate(policy_state)
    

    dataset = MySFTDataset(
        DATASET[args.task.query_dataset],
        tokenizer,
        args.task.query_length,
        seed=local_seed,
        start_text=args.task.start_text,
        end_text=args.task.end_text,
    )
    
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.sft.batch_size,
        collate_fn=numpy_collate,
    )
    iter_dataloader = iter(dataloader)
    

    def train_update(policy_state, input_ids, response_ids, sft_stats):
        queries = right_padding_to_left_padding(input_ids, tokenizer.pad_token_id)

        query_responses = jnp.concatenate((input_ids, response_ids), axis=1)

        def sft_single_microbatch(carry, inp):
            policy_state, sft_stats = carry
            mb_query_responses = inp

            policy_state, sft_stats = train_step(
                policy_state=policy_state,
                sft_stats=sft_stats,
                mb_query_responses=mb_query_responses,
                tokenizer=tokenizer,
                args=args
            )
            return (policy_state, sft_stats), None

        def sft_single_epoch(carry, inp):
            policy_state, sft_stats, key = carry
            key, subkey = jax.random.split(key, 2)
            perm = jax.random.permutation(key, args.sft.local_batch_size)
            # That is -> query_responses, logprobs = inp
            
            mbs_query_responses = einops.rearrange(
                query_responses[perm],
                "(c m) l -> c m l",
                c=args.sft.gradient_accumulation_steps,
            )
            
            (policy_state, sft_stats), _ = jax.lax.scan(
                f=sft_single_microbatch,
                init=(policy_state, sft_stats),
                xs=(
                    mbs_query_responses
                ),
            )
            return (policy_state, sft_stats, key), None

        key = jax.random.PRNGKey(args.seed)
        # Do multiple epochs of sft training, with a fresh random shuffle in each epoch
        (policy_state, sft_stats, _), _ = jax.lax.scan(
            f=sft_single_epoch,
            init=(policy_state, sft_stats, key),
            xs=None,
            length=args.sft.noptepochs,
        )

        sft_stats = jax.lax.pmean(sft_stats.compute(), "batch")

        samples_to_print = dict(
            query_response=query_responses[0]
        )
        return policy_state, sft_stats, samples_to_print

    p_train_update = jax.pmap(
        train_update,
        axis_name="batch",
        donate_argnums=(0,),
    )
    
    global_step = 0
    
    for update, [input_ids, response_ids] in tqdm(enumerate(iter_dataloader)):
        global_step += args.sft.batch_size
        input_ids = common_utils.shard(input_ids)
        response_ids = common_utils.shard(response_ids)
        sft_stats = jax_utils.replicate(SFTStatistics.empty())
        policy_state, sft_stats, samples_to_print = p_train_update(
            policy_state=policy_state,
            input_ids=input_ids,
            response_ids = response_ids,
            sft_stats=sft_stats
        )
        
        # save model
        if (args.local_rank == 0) and (update%1000==0):
            if args.save_path:
                ckpt = {"policy_model": jax_utils.unreplicate(policy_state), "args": vars(args)}
                orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
                save_args = orbax_utils.save_args_from_target(ckpt)
                orbax_checkpointer.save(args.save_path+"model_"+update+"/", ckpt, save_args=save_args, force=True)

            if args.local_rank == 0 and args.track:
                wandb.finish()

        # try:
        #   sample_query_response = samples_to_print["query_response"][0]
        #     console.print(
        #         f"[green][bold]{'Query'}:[/]\n"
        #         + f"[green]{ tokenizer.decode(sample_query_response[:args.task.query_length], skip_special_tokens=True)}[/]\n\n"
        #         + f"[yellow][bold]{'Response'}:[/]\n"
        #         )
        # except Exception as e:
        #     print(e)

        
        # RL metrics aggregated at the batch level
        sft_stats = common_utils.get_metrics([sft_stats])
        writer.add_scalar("sft/loss/total", sft_stats["loss"].item(), update)

        # Logging learning rate and learning progress
        writer.add_scalar("sft/lr", policy_state.opt_state.inner_opt_state.hyperparams["learning_rate"][0].item(), update)
        writer.add_scalar("sft/episode", global_step, update)

    
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
    train(args)
