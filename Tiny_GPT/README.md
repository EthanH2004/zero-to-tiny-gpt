# Tiny GPT

A small, character-level GPT built in PyTorch for learning how language
models are structured, trained, saved, and used as an interactive chatbot.

## First-time setup

```bash
python3 -m venv .venv
./.venv/bin/python -m pip install -r requirements.txt
```

The project can then be controlled from one menu:

```bash
./.venv/bin/python main.py
```

Choose option **7** to run the complete V2 pipeline unattended overnight.
Choose option **8** after it finishes to start the V2 chatbot. V2 lives in its
own folder and does not replace the completed character-level V1 model.

The explicit interpreter path works even when a parent folder contains a
colon, which would otherwise split the shell's `PATH` during activation.

## Complete workflow

Run these commands in order from the `Tiny_GPT` directory:

```bash
# 1. Download and prepare TinyStories.
./.venv/bin/python prepare_data.py

# 2. Train the base language model and save checkpoints/base.pt.
./.venv/bin/python train.py

# 3. Test ordinary text generation from the base model.
./.venv/bin/python generate.py

# 4. Download and prepare OpenAssistant dialogue.
./.venv/bin/python prepare_chat_data.py

# 5. Continue training the base model for conversation.
#    This saves checkpoints/chat.pt without replacing base.pt.
./.venv/bin/python train.py --fine-tune

# 6. Start an interactive conversation.
./.venv/bin/python chatbot.py
```

The preparation scripts skip files that already exist. Pass `--force` to
either preparation script when you intentionally want to recreate its data.

## What each file does

- `config.toml` contains the settings for data, model size, training,
  generation, fine-tuning, and chat.
- `main.py` provides one menu for running the complete workflow.
- `settings.py` loads that configuration.
- `tiny_gpt.py` defines the tokenizer, attention heads, transformer blocks,
  GPT model, generation method, and checkpoint functions.
- `prepare_data.py` builds the reproducible TinyStories base dataset.
- `train.py` trains the base model or fine-tunes it with `--fine-tune`.
- `generate.py` loads `base.pt` and continues a prompt.
- `prepare_chat_data.py` builds assistant dialogue from OpenAssistant OASST1.
- `chatbot.py` loads `chat.pt` and runs the interactive chat loop.
- `tests/` checks the reusable model code automatically.

## Useful overrides

The normal settings live in `config.toml`, but individual runs can override
them:

```bash
./.venv/bin/python train.py --steps 1000
./.venv/bin/python generate.py --prompt "once upon a time" --temperature 0.7
./.venv/bin/python chatbot.py --temperature 0.6
```


The generated datasets and model checkpoints are intentionally excluded from
Git because they can be recreated by the commands above.
