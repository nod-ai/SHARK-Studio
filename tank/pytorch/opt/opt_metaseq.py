import json
import os
import os.path
import random
import string

from metaseq import options
from metaseq.dataclass.utils import convert_namespace_to_omegaconf
from metaseq_cli.train import main

try:
    import tokenizers  # noqa

    has_hf_tokenizers = True
except ImportError:
    has_hf_tokenizers = False


def write_one_jsonl_(jsonl_path, num_lines=5, text_len_min=5, text_len_max=50):
    data = []
    with open(jsonl_path, "w") as h:
        for _ in range(num_lines):
            text_len = random.choice(range(text_len_min, text_len_max))
            data.append(
                {"text": "".join(random.choices(string.ascii_letters, k=text_len))}
            )
            print(json.dumps(data[-1]), file=h)
    return


def write_dummy_jsonl_data_dir_(data_dir, num_lines=500):
    for subset in ["train", "valid"]:
        for shard in range(2):
            shard_dir = os.path.join(data_dir, subset, f"{shard:02}")
            os.makedirs(shard_dir)
            for dataset in ["a", "b"]:
                write_one_jsonl_(
                    os.path.join(shard_dir, f"dataset_{dataset}.jsonl"),
                    num_lines=num_lines,
                )


def write_dummy_bpe_(data_dir):
    from tokenizers import ByteLevelBPETokenizer

    tokenizer = ByteLevelBPETokenizer(add_prefix_space=True)
    tokenizer.train(
        [],
        vocab_size=100,
        special_tokens=["<s>", "<pad>", "</s>", "<unk>"],
        show_progress=False,
    )
    vocab, merges = tokenizer.save_model(data_dir)
    return vocab, merges


data_dir = os.path.curdir + "/tests"

assert not os.path.exists(data_dir), f"Please remove directory {data_dir}"

write_dummy_jsonl_data_dir_(data_dir)
vocab, merges = write_dummy_bpe_(data_dir)

train_parser = options.get_training_parser()
train_args = options.parse_args_and_arch(
    train_parser,
    [
        "--task",
        "streaming_language_modeling",
        data_dir,
        "--arch",
        "transformer_lm_gpt",
        "--optimizer",
        "adam",
        "--lr",
        "0.0001",
        "--lr-scheduler",
        "inverse_sqrt",
        "--tokens-per-sample",
        "100",
        "--save-dir",
        data_dir,
        "--max-epoch",
        "1",
        "--distributed-world-size",
        "1",
        "--ddp-backend",
        "pytorch_ddp",
        "--num-workers",
        "0",
    ]
    + [
        "--vocab-filename",
        f"{data_dir}/vocab.json",
        "--merges-filename",
        f"{data_dir}/merges.txt",
        "--dropout",
        "0.0",
        "--log-format",
        "json",
        "--log-interval",
        "1",
        "--max-epoch",
        "1",
        "--batch-size",
        "2",
        "--save-interval-updates",
        "3",
    ],
)
cfg = convert_namespace_to_omegaconf(train_args)

main(cfg)
