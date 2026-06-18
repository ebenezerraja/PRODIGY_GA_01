# PRODIGY_GA_01 — Text Generation with GPT-2

## What this task is about
GPT-2 is a **transformer decoder** trained on the task of **causal language
modeling**: given all the tokens so far, predict the next one. "Fine-tuning"
takes a GPT-2 checkpoint that already knows general English and continues
training it on a much smaller, specific dataset, nudging its predicted
probability distribution to favor the style, vocabulary, and structure of
that dataset. The architecture doesn't change — only the weights shift.

## How the code works
- `train.py` loads `gpt2` (124M params) and its tokenizer, chops your text
  file into fixed-length token blocks, and fine-tunes with Hugging Face's
  `Trainer` using the standard "shift-by-one" causal LM loss (cross-entropy
  between predicted next-token distribution and the actual next token).
- `generate.py` loads the fine-tuned checkpoint and samples new text from a
  prompt using temperature + top-k + top-p sampling (this is what makes
  output varied instead of always picking the single most likely word).

## Files
- `train.py`, `generate.py`
- `sample_data.txt` — 30 short, single-sentence "engineering aphorisms" I
  wrote as a tiny demo dataset. It's intentionally small and stylistically
  consistent so that after a few epochs you can clearly see the model
  picking up the "short aphorism" pattern. **Swap this for your own larger
  text file for a real submission** — a collection of your own notes, a
  public-domain book, your own blog posts, etc. More data = less
  memorization, more genuine style transfer.

## Running it
This needs internet access (to download the GPT-2 weights) and ideally a
GPU — I can't execute this step in this sandboxed environment since it has
no internet access, so run it on **Google Colab** (free GPU):

```bash
pip install -r requirements.txt
python train.py --data sample_data.txt --epochs 3 --output_dir ./gpt2-finetuned
python generate.py --model_dir ./gpt2-finetuned --prompt "The best dataset"
```

Compare against the un-fine-tuned base model to see the difference:
```bash
python generate.py --model_dir gpt2 --prompt "The best dataset"
```

## What to expect
With only 30 short lines, the model will lean toward memorizing those exact
sentences rather than truly generalizing — that's normal and worth
mentioning in your write-up. With a few hundred KB+ of consistent-style
text, you'll see genuine new sentences that *feel* like the source material
without repeating it verbatim.

## For your LinkedIn post / report
Mention: the causal LM objective, why fine-tuning is cheaper than training
from scratch (you're starting from weights that already understand
grammar/world knowledge), and the coherence/diversity tradeoff controlled
by `temperature`, `top_k`, and `top_p` at generation time.
