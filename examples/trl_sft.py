from typing import cast

from datasets import Dataset, load_dataset
from trl import SFTConfig, SFTTrainer

# get dataset
dataset = cast(
    Dataset, load_dataset("tiny_shakespeare", split="train", trust_remote_code=True)
)

config = SFTConfig(
    output_dir="output",
    dataset_text_field="text",
    max_seq_length=512,
)

# get trainer
trainer = SFTTrainer("openai-community/gpt2", train_dataset=dataset, args=config)

# train
trainer.train()  # type: ignore
