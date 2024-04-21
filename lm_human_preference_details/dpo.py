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
    
    eval_batch_size: int = 32
    eval_accum_steps: int = 4
    local_micro_batch_size: tyro.conf.Suppress[int] = None
    """per rank micro batch size"""
    world_size: tyro.conf.Suppress[int] = None
    batch_size: tyro.conf.Suppress[int] = None
    minibatch_size: tyro.conf.Suppress[int] = None
    num_updates: tyro.conf.Suppress[int] = None
    nminibatches: int = 1
    
    # Learning rate, epochs, episodes
    opt_choice = "rmsprop"
    
    total_episodes: int = 118000
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
    train: DPOParams = field(default_factory=DPOParams)

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



@flax.struct.dataclass
class LMBackboneParams:
    """Parameters for the language model backbone."""

    lm_backbone_params: flax.core.FrozenDict


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
        
        dpo_loss = -flax.linen.log_sigmoid(args.train.beta*temp)
        
        dpo_loss_val = jnp.mean(dpo_loss)

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

def train(args: Args):
    local_devices = jax.local_devices()
    global_devices = jax.devices()
    args.train.world_size = jax.process_count()
    learner_devices = [local_devices[d_id] for d_id in args.learner_device_ids]
    global_learner_devices = [
        global_devices[d_id + process_index * len(local_devices)]
        for process_index in range(args.train.world_size)
        for d_id in args.learner_device_ids
    ]
    pprint({"global_learner_devices": global_learner_devices})
    args.global_learner_devices = [str(item) for item in global_learner_devices]
    args.learner_devices = [str(item) for item in learner_devices]
    args.local_rank = jax.process_index()
    args.train.batch_size = int(args.train.local_batch_size * len(args.learner_devices) * args.train.world_size)
    args.train.minibatch_size = exact_div(args.train.batch_size, args.train.nminibatches)
    args.train.local_mini_batch_size = exact_div(args.train.local_batch_size, args.train.nminibatches)
    args.train.local_micro_batch_size = exact_div(args.train.local_mini_batch_size, args.train.gradient_accumulation_steps)
    # `per_rank_rollout_batch_size` is our `args.train.local_batch_size`
    # `per_rank_minibatch_size` is our `args.train.local_mini_batch_size`
    args.train.num_updates = args.train.total_episodes // args.train.batch_size

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

    if (args.train.opt_choice=="adam_tf"):
        optim_choice = adam_tf_style
    elif (args.train.opt_choice=="rmsprop"):
        optim_choice = optax.rmsprop
    elif (args.train.opt_choice=="adamw"):
        optim_choice = optax.adamw
    else:
        optim_choice = optax.adam
    

    optimizer = optax.MultiSteps(
        optax.inject_hyperparams(optim_choice)(
            learning_rate = functools.partial(linear_warmup_schedule, args=args), 
            eps=args.train.eps,
        ),
        every_k_schedule=args.train.gradient_accumulation_steps,
    )
    
    print("optimizer initialized")

    policy_state = TrainState.create(apply_fn=policy_forward, params=policy_params, tx=optimizer)
    policy_state = jax_utils.replicate(policy_state)
    
    print("Train state created")

    train_dataloader = get_batch_loader(tokenizer, args, seed=local_seed, split='train')
    eval_dataloader = get_batch_loader_eval(tokenizer, args, seed=local_seed, split='validation[:6000]')

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
        batch_size=args.train.batch_size,
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
            perm = jax.random.permutation(key, args.train.local_batch_size)
            # That is -> query_responses, logprobs = inp
            
            # For both rejected and preferred query responses
            mbs_query_responses_pref = einops.rearrange(
                query_responses_pref[perm],
                "(c m) l -> c m l",
                c=args.train.gradient_accumulation_steps,
            )

            mbs_query_responses_rej = einops.rearrange(
                query_responses_rej[perm],
                "(c m) l -> c m l",
                c=args.train.gradient_accumulation_steps,
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
            length=args.train.noptepochs,
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

        def dpo_single_microbatch_eval(inp1, inp2):
            mb_query_responses_pref, mb_query_responses_rej = inp1, inp2

            loss_val = eval_step(
                policy_state=policy_state,
                ref_model=ref_model,
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
                c=args.train.eval_accum_steps,
        )

        mbs_query_responses_rej = einops.rearrange(
                query_responses_rej,
                "(c m) l -> c m l",
                c=args.train.eval_accum_steps,
        )

        eval_epoch = jax.vmap(dpo_single_microbatch_eval, in_axes=0)
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

    
        global_step += args.train.batch_size
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
        
        if (args.local_rank == 0) and (update>0) and (update%args.save_every==0): # (update>0) and (update%1000==0):
            if args.save_path:
                ckpt = {"policy_model": jax_utils.unreplicate(policy_state), "args": vars(args)}
                orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
                save_args = orbax_utils.save_args_from_target(ckpt)
                orbax_checkpointer.save(args.save_path + f"model_{update}/", ckpt, save_args=save_args, force=True)

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
