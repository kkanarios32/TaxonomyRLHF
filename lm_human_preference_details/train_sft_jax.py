from torch.utils.data import IterableDataset
from transformers import AutoTokenizer, FlaxAutoModelForCausalLM, GenerationConfig
from lm_human_preference_details.data import DATASET

print("imports done")

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
        for query, response in self.generator("train", self.seed, shuffle=True):
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
                padding=False,
                max_length=self.query_length,
                return_tensors="np",
                return_attention_mask=False,
            )

            max_length = self.tokenizer.model_max_length - self.query_length
            response_output = self.tokenizer(response,
                                             max_length=max_length,
                                             truncation=True)

            yield query_output["input_ids"], response_output["input_ids"]

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
        530,
        seed=12,
        start_text=None,
        end_text=None,
    )

print("dataset generated")
dataset = iter(dataset)

print("dataset iterable created")
query_shapes=[]

for i in range(100):
    query, response = next(dataset)
    query_shapes.append(query.shape[0])

print(max(query_shapes), sum(query_shapes)/len(query_shapes))
