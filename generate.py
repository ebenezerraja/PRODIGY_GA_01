"""
PRODIGY_GA_01 - Generate text with a fine-tuned (or vanilla) GPT-2 model.

Usage:
    python generate.py --model_dir ./gpt2-finetuned --prompt "The future of AI"
    python generate.py --model_dir gpt2 --prompt "Once upon a time"   # vanilla GPT-2
"""

import argparse
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast


def generate(model_dir, prompt, max_length=80, temperature=0.9, top_k=50,
             top_p=0.95, num_return_sequences=1):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = GPT2TokenizerFast.from_pretrained(model_dir)
    model = GPT2LMHeadModel.from_pretrained(model_dir).to(device)
    model.eval()

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_length=max_length,
            do_sample=True,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            num_return_sequences=num_return_sequences,
            pad_token_id=tokenizer.eos_token_id,
        )

    return [tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="./gpt2-finetuned",
                         help="Path to a fine-tuned model dir, or a hub id like 'gpt2'")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--max_length", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--num_return_sequences", type=int, default=1)
    args = parser.parse_args()

    outputs = generate(
        args.model_dir, args.prompt,
        max_length=args.max_length,
        temperature=args.temperature,
        num_return_sequences=args.num_return_sequences,
    )

    for i, text in enumerate(outputs, 1):
        print(f"\n--- Sample {i} ---\n{text}\n")


if __name__ == "__main__":
    main()
