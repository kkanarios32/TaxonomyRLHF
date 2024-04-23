import random
import re

import ftfy
import json
import os
from datasets import load_dataset
from tqdm import tqdm

dir = os.path.dirname(__file__)
dir = os.path.join(dir, "datasets")

def tldr_filtered_sft_generator(split, seed=0, shuffle=False):
    assert split in ["test", "train", "valid"]
    file = os.path.join(dir, "tldr", f"{split}.jsonl")

    with open(file) as f:
        datas = list(f)
    
    if shuffle:
        random.seed(seed)
        random.shuffle(datas)

    for data in datas:
        data = json.loads(data)
        subreddit = "SUBREDDIT: r/" + data['subreddit']
        title = "\n\nTITLE: " + data['title']
        post = "\n\nPOST: " + data['post'] + "\n\nTL;DR:"
        query = subreddit + title + post
        summary = data['summary']
        yield query, summary
        
def fam_rel_sft_generator(split, seed=0, shuffle=False):
    assert split in ["eval", "train"]
    file = os.path.join(dir, "fam_rel", f"fam_sft_{split}_data.jsonl")

    with open(file) as f:
        datas = list(f)
    
    if shuffle:
        random.seed(seed)
        random.shuffle(datas)

    for data in datas:
        data = json.loads(data)
        yield data["query"], data["response"]
        
def fam_rel_shuffled_resp_sft_generator(split, seed=0, shuffle=False):
    assert split in ["eval", "train"]
    file = os.path.join(dir, "fam_rel", f"fam_sft_{split}_data_shuffled.jsonl")

    with open(file) as f:
        datas = list(f)
    
    if shuffle:
        random.seed(seed)
        random.shuffle(datas)

    for data in datas:
        data = json.loads(data)
        yield data["query"], data["response"]
        
def fam_rel_shuffled_resp_3_children_sft_generator(split, seed=0, shuffle=False):
    assert split in ["eval", "train"]
    file = os.path.join(dir, "fam_rel", f"fam_sft_{split}_data_shuffled_3_children.jsonl")

    with open(file) as f:
        datas = list(f)
    
    if shuffle:
        random.seed(seed)
        random.shuffle(datas)

    for data in datas:
        data = json.loads(data)
        yield data["query"], data["response"]
        
        
def tldr_kto_random_generator(split="train", seed=0, shuffle=False): 
    """
    Generator for DPO. Outputs two different summaries: preferred and rejected.
    """

    assert split in ["train", "validation[:6000]"]

    datas = load_dataset('openai/summarize_from_feedback',
                       'comparisons', 
                       split=f'{split}',
                       streaming=False,
    ).select_columns(['info', 'summaries', 'choice'])
    
    # This gives errors for IterableDatasets
    if shuffle:
        random.seed(seed)
        random.shuffle(datas)

    for data in datas:
        # Don't need below
        # data = json.loads(data)
        choice = random.randint(0,1)
        subreddit = "SUBREDDIT: r/" + data['info']['subreddit']
        title = "\n\nTITLE: " + data['info']['title']
        post = "\n\nPOST: " + data['info']['post'] + "\n\nTL;DR:"
        query = subreddit + title + post

        # For two different summaries
        summary = data['summaries'][choice]['text']
        chosen_label = True if (choice==data['choice']) else False

        yield query, summary, chosen_label


def tldr_dpo_generator(split="train", seed=0, shuffle=False): 
    """
    Generator for DPO. Outputs two different summaries: preferred and rejected.
    """

    assert split in ["train", "validation[:6000]"]

    datas = load_dataset('openai/summarize_from_feedback',
                       'comparisons', 
                       split=f'{split}',
                       streaming=False,
    ).select_columns(['info', 'summaries', 'choice'])
    
    # This gives errors for IterableDatasets
    if shuffle:
        random.seed(seed)
        random.shuffle(datas)

    for data in datas:
        # Don't need below
        # data = json.loads(data)
        subreddit = "SUBREDDIT: r/" + data['info']['subreddit']
        title = "\n\nTITLE: " + data['info']['title']
        post = "\n\nPOST: " + data['info']['post'] + "\n\nTL;DR:"
        query = subreddit + title + post

        # For two different summaries
        summary_prefer = data['summaries'][data['choice']]['text']
        summary_reject = data['summaries'][data['choice']-1]['text']

        yield query, summary_prefer, summary_reject


# bookcorpus dataset, modified from
# https://github.com/openai/lm-human-preferences/blob/cbfd210bb8b08f6bc5c26878c10984b90f516c66/lm_human_preferences/datasets/books.py
def books_generator(mode, seed=0, shuffle=False):
    dataset = load_dataset("bookcorpus", split=mode)

    if shuffle:
        random.seed(seed)
        dataset = dataset.shuffle(seed)

    while True:
        for _, data in enumerate(dataset):
            text = data["text"]
            yield text


# Cnn_dailymail dataset, modified from
# https://github.com/openai/lm-human-preferences/blob/cbfd210bb8b08f6bc5c26878c10984b90f516c66/lm_human_preferences/datasets/cnndm.py
def clean_up_start(text):
    text = re.split(r"\(CNN\) +--", text)[-1]
    text = re.split(r"\(CNN\)", text[:100])[-1] + text[100:]
    text = re.sub(r"^and \w+\n", "", text)
    text = re.split(r".*UPDATED:\s+[0-9]{2}:[0-9]{2}.*[2011|2012|2013|2014|2015]", text)[-1]
    text = text.replace("’", "'")
    text = text.replace("‘", "'")
    return text.strip()


def cnndm_generator(mode, seed=0, shuffle=False):
    dataset = load_dataset("cnn_dailymail", version="3.0.0", split=mode)

    if shuffle:
        random.seed(seed)
        dataset = dataset.shuffle(seed)

    for _, data in enumerate(dataset):
        original_text = data["article"]
        text = clean_up_start(original_text)
        text = ftfy.fix_text(text)

        text = re.sub(r"\n{3,}", "\n\n", text)
        text = text.split("@highlight")[0].strip()

        yield text


# for testing only
def dummy_generator(mode, seed=0, shuffle=False):
    while True:
        yield "dummy"


DATASET = {
    "books": books_generator,
    "cnndm": cnndm_generator,
    "tldr-sft": tldr_filtered_sft_generator,
    "fam-rel-sft": fam_rel_sft_generator,
    "fam-rel-sft-shuffled": fam_rel_shuffled_resp_sft_generator,
    "fam-rel-sft-shuffled-3-children": fam_rel_shuffled_resp_3_children_sft_generator,
    "tldr-dpo": tldr_dpo_generator,
    "tldr-kto-random": tldr_kto_random_generator,
    "dummy": dummy_generator,
}
