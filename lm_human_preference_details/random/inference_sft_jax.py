print("started")
from jax.lib import xla_bridge
print(xla_bridge.get_backend().platform)

from torch.utils.data import IterableDataset
from transformers import AutoTokenizer, FlaxAutoModelForCausalLM, GenerationConfig
from lm_human_preference_details.data import DATASET
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp

num_resp_start =2009
num_resp_end=2008
num_repeat = 1
query_len = 600
resp_len = 424
temp = 0.7

print("imports done")

# a pytorch dataset
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
        for query, response in self.generator("test", self.seed, shuffle=True):
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

            max_response_length = resp_len
            response_tokens = self.tokenizer.encode(response, max_length=max_response_length,
                                                 truncation=True)
            response_output = self.tokenizer.pad({"input_ids": response_tokens},
                                                 padding="max_length",
                                                 max_length=max_response_length,
                                                 return_tensors = "np",
                                                 return_attention_mask=False
                                                )

            yield query_output["input_ids"], np.squeeze(response_output["input_ids"]), query, response

print("read the function")
tokenizer = AutoTokenizer.from_pretrained(
        "gpt2",
        padding_side="right",
    )
tokenizer.add_special_tokens({"pad_token": "[PAD]"})

print("tokenizer initialized")

dataset = MySFTDataset(
        DATASET["tldr-sft"],
        tokenizer,
        query_len,
        seed=10,
        start_text=None,
        end_text=None,
    )

print("dataset generated")
dataset = iter(dataset)

print("dataset iterable created")
query_shapes=[]
# i=0
# for tup in dataset:
#     query_shapes.append(len(tup[0]))
#     i +=1
#     if (i==60000):
#         break
# print(max(query_shapes))


lm_backbone = FlaxAutoModelForCausalLM.from_pretrained("kwonsm/gpt2-tldr-dpo-updated")
lm_backbone_2 = FlaxAutoModelForCausalLM.from_pretrained("kkanarios/gpt2-tldr-sft")
lm_backbone_3 = FlaxAutoModelForCausalLM.from_pretrained("kwonsm/gpt2-tldr-kto")

# orbax_checkpointer = ocp.PyTreeCheckpointer()
# sft_model = orbax_checkpointer.restore('/scratch/tewaria_root/tewaria0/ckausik/sftmodels-fam-rel-shuffled-3-children/final_model/')['policy_model']['params']['lm_backbone_params']['params']

# print(dpo_model.keys())

# sft_model = FlaxAutoModelForCausalLM.from_pretrained("kkanarios/gpt2-tldr-sft").params

# # disable `pad_token_id` and `eos_token_id` because we just want to
# # generate tokens without truncation / padding
lm_backbone.generation_config.eos_token_id = None
lm_backbone.generation_config.pad_token_id = tokenizer.pad_token_id

generation_config = GenerationConfig(
    max_new_tokens=resp_len,
    temperature=temp,
    top_k=0.0,
    top_p=1.0,
    do_sample=True,
    pad_token_id=tokenizer.pad_token_id,
)

def policy_forward(
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
        variables={"params": lm_backbone.params},
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids
    )

    # shape: [batch_size, length, 1]
    return lm_backbone_out

def policy_generate(
        model_params,
        queries: jnp.ndarray,
    ):
        input_ids = queries
        attention_mask = input_ids != tokenizer.pad_token_id
        input_ids = jnp.where(attention_mask, queries, 0)
        output = lm_backbone.generate(
            params=model_params,
            input_ids=input_ids,
            generation_config=generation_config,
            attention_mask=attention_mask.astype("i4"),
            return_dict_in_generate=True,
        )
        query_length = input_ids.shape[1]
        return jnp.concatenate((queries, output.sequences[:, query_length:]), axis=1)

print("lm backbone created")
i=0
# print(tokenizer(" "))
for elem in dataset:
    query, response, query_words, response_words = elem
    i+=1
    if i>=num_resp_start:
        print("\n Response num:", i)
        print(query_words, "\n")
        print("Human: ", response_words, "\n")
        for j in range(num_repeat):
            
            
            gen_response_sft = policy_generate(lm_backbone_2.params, np.reshape(query, (1, query_len)))
            # print(gen_response_sft[0, query_len:])
            print("SFT:", tokenizer.decode(gen_response_sft[0, query_len:]))
            
            gen_response_sft = policy_generate(lm_backbone.params, np.reshape(query, (1, query_len)))
            # print(gen_response_sft[0, query_len:])
            print("DPO:", tokenizer.decode(gen_response_sft[0, query_len:]), "\n")
            
            gen_response_sft = policy_generate(lm_backbone_3.params, np.reshape(query, (1, query_len)))
            # print(gen_response_sft[0, query_len:])
            print("KTO:", tokenizer.decode(gen_response_sft[0, query_len:]), "\n")
            
        
    if i > num_resp_end:
        break
        
        
# new_query_tokens = tokenizer.encode("What is the capital of Hungary?")
# query = tokenizer.pad({"input_ids": new_query_tokens},
#                                                  padding="max_length",
#                                                  max_length=600,
#                                                  return_tensors = "np",
#                                                  return_attention_mask=False
#                                                 )["input_ids"]
    
# query, response, query_words, response_words = dataset[1]
# query = np.reshape(query, (1,query_len))
# response = np.reshape(response, (1,resp_len))
# query_response = jnp.concatenate((query, response), axis=1)

# print("query response processed")
# print(response[:resp_len])
# logits = policy_forward(query_response).logits[:, 600:, :]
# print("logits made")
# response_logprobs = -optax.softmax_cross_entropy_with_integer_labels(logits, response)
# filter_for_pad_logprobs = (response!=tokenizer.pad_token_id)
# response_logprobs=response_logprobs*filter_for_pad_logprobs
# print("cross entropy taken, logprobs filtered")

# # print(response_logprobs)
# sft_loss_val = -jnp.sum(response_logprobs)
# print(sft_loss_val)

# print(query_words)
# print(response_words)

# print("what is the capital of Hungary?")

# gen_response_sft = policy_generate(sft_model, query)

# new_response = gen_response[:, query_len:]
# # print(type(gen_response), (np.array(gen_response)).shape)
# # logits = policy_forward(gen_response).logits[:, 600:, :]
# # print("logits made again")
# # new_response_logprobs = -optax.softmax_cross_entropy_with_integer_labels(logits, new_response)
# # filter_for_pad_logprobs = (new_response!=tokenizer.pad_token_id)
# # new_response_logprobs=new_response_logprobs*filter_for_pad_logprobs
# # print("cross entropy taken again, logprobs filtered again")

# # print(new_response_logprobs)
# sft_loss_val = -jnp.sum(new_response_logprobs)
# print(sft_loss_val)

# print("new dpo response:")
# print(tokenizer.decode(gen_response_dpo[0, 600:]))

# print("new sft response:")
# gen_response_sft = policy_generate(sft_model, query)

# print(tokenizer.decode(gen_response_sft[0, query_len:]))
