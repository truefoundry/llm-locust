import json
import os

import requests
from transformers import PreTrainedTokenizerBase

system_prompt = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""


def get_prompt_set(
    tokenizer: PreTrainedTokenizerBase,
    min_input_length: int = 0,
    max_input_length: int = 500,
) -> list[dict]:
    """
    Return a list of prompts with length between min_input_length and max_input_length
    """
    # check if the dataset is cached
    if os.path.exists("databricks-dolly-15k.jsonl"):
        print("Loading cached dataset")
        with open("databricks-dolly-15k.jsonl") as f:
            dataset = [json.loads(line) for line in f.readlines()]
    else:
        print("Downloading dataset")
        raw_dataset = requests.get(
            "https://huggingface.co/datasets/databricks/databricks-dolly-15k/resolve/main/databricks-dolly-15k.jsonl",
            timeout=60,
        )
        content = raw_dataset.content
        with open("databricks-dolly-15k.jsonl", "wb") as f:
            f.write(content)
        dataset = [json.loads(line) for line in content.decode().split("\n")]
        print("Dataset downloaded")

    for d in dataset:
        user_prompt = d["context"] + d["instruction"]
        chat = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        d["question"] = user_prompt
        d["num_input_tokens"] = len(
            tokenizer.apply_chat_template(
                chat, tokenize=True, add_generation_prompt=True
            )
        )

    return [
        {"prompt": d["question"], "num_input_tokens": d["num_input_tokens"]}
        for d in dataset
        if min_input_length <= d["num_input_tokens"] <= max_input_length
    ]


def get_prompt_set_random(tokenizer: PreTrainedTokenizerBase) -> list[dict]:
    """
    Return a list of prompts randomly generated by genai-perf
    """
    with open("inputs.json") as f:
        dataset = json.loads(f.read())["data"]

    for d in dataset:
        chat = [
            {"role": "user", "content": d["payload"][0]["prompt"][0]},
        ]
        d["question"] = d["payload"][0]["prompt"][0]
        d["num_input_tokens"] = len(
            tokenizer.apply_chat_template(
                chat, tokenize=True, add_generation_prompt=True
            )
        )

    return [
        {"prompt": d["question"], "num_input_tokens": d["num_input_tokens"]}
        for d in dataset
    ]


def get_prompt_set_single(tokenizer: PreTrainedTokenizerBase) -> list[dict]:
    """
    Return a single random prompt
    """
    prompt = """hi hi hi have been a uniform sacrifice of inclination to the
        spirit of criticism, the constancy of your support was
        conviction that the step is compatible with both.
        feelings do not permit me to suspend the deep
        administer the executive government of the United
        tender of service which silence in my situation might
        among the number of those out of whom a choice is to be made.
        admonishes me more and more that the shade of
        public voice, that I should now apprise you of the
        confidence with which it has supported me; and for
        the best exertions of which a very fallible judgment
        to be your desire. I constantly hoped that it would
        penetrated with this idea, I shall carry it with me to
        If benefits have resulted to our country from these
        arrived when your thoughts must be employed in
        the preparation of an address to declare it to you; but
        I rejoice that the state of your concerns, external as
        states, under the auspices of liberty, may be made
        acknowledgment of that debt of gratitude which I
        given peculiar value to my services, they were
        and adoption of every nation which is yet a stranger to it.
        retirement is as necessary to me as it will be
        have, with good intentions, contributed towards the
        every direction were liable to mislead, amidst
        do this, previous to the last election, had even led to
        citizen to his country\u2014and that, in withdrawing the
        organization and administration of the government,  In the discharge of this """
    dataset = [{"content": prompt * 3}]
    for d in dataset:
        chat = [
            {"role": "user", "content": d["content"]},
        ]
        d["question"] = d["content"]
        d["num_input_tokens"] = len(
            tokenizer.apply_chat_template(
                chat, tokenize=True, add_generation_prompt=True
            )
        )

    return [
        {"prompt": d["question"], "num_input_tokens": d["num_input_tokens"]}
        for d in dataset
    ]
