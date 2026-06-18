"""
PRODIGY_GA_01 - Text Generation with GPT-2 (fine-tuning)
------------------------------------------------------------
Fine-tunes a pretrained GPT-2 checkpoint (default: "gpt2", 124M params) on
a custom text file using causal language modeling, so generated text
mimics the style/structure of that file.

NOTE: this needs internet access (to download the pretrained weights) and
ideally a GPU. Run it on Google Colab: Runtime > Change runtime type > GPU.
It will also run on CPU, just much more slowly.

Usage:
    python train.py --data sample_data.txt --epochs 3 --output_dir ./gpt2-finetuned
"""

import argparse
import torch
from torch.utils.data import Dataset
from transformers import (
    GPT2LMHeadModel,
    GPT2TokenizerFast,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)


class BlockTextDataset(Dataset):
    """Reads a text file and chops its tokens into fixed-length blocks."""

    def __init__(self, tokenizer, file_path: str, block_size: int = 128):
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        token_ids = tokenizer.encode(text)
        self.examples = [
            token_ids[i:i + block_size]
            for i in range(0, max(1, len(token_ids) - block_size + 1), block_size)
        ]
        if not self.examples:
            self.examples = [token_ids[:block_size]]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return torch.tensor(self.examples[idx], dtype=torch.long)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune GPT-2 on a custom text file")
    parser.add_argument("--data", type=str, default="sample_data.txt")
    parser.add_argument("--model_name", type=str, default="gpt2")
    parser.add_argument("--output_dir", type=str, default="./gpt2-finetuned")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    args = parser.parse_args()

    tokenizer = GPT2TokenizerFast.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = GPT2LMHeadModel.from_pretrained(args.model_name)

    dataset = BlockTextDataset(tokenizer, args.data, block_size=args.block_size)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        save_strategy="epoch",
        logging_steps=10,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print(f"Done. Fine-tuned model saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
