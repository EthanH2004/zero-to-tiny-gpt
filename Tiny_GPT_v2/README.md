# Tiny GPT V2

V2 keeps the complete V1 project intact and adds:

- a 2,048-piece byte-level BPE tokenizer instead of character tokens;
- a 15.86-million-parameter, eight-layer GPT;
- 100 million characters of FineWeb-Edu base training data;
- complete multi-turn OpenAssistant conversation paths;
- time-budgeted training, cosine learning-rate decay, validation, best-model
  preservation, periodic recovery checkpoints, and early stopping;
- one unattended overnight pipeline.

Start the root `main.py` control center and choose option 7. When the pipeline
finishes, choose option 8 to chat with V2.

The configured ceilings are six hours for base training and 75 minutes for
dialogue fine-tuning. A real-device benchmark measured about 8.1 steps per
second before prolonged thermal load, putting the complete run under
approximately eight hours on the target Mac. Step limits and early stopping can
finish sooner when additional repetition would no longer help validation loss.
