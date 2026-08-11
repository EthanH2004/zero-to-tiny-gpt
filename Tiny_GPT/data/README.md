# Training data

Run `python prepare_data.py` to recreate the TinyStories base-training files.
TinyStories accompanies the Microsoft Research paper *TinyStories: How Small
Can Language Models Be and Still Speak Coherent English?* and is distributed
under the CDLA-Sharing-1.0 license.

Run `python prepare_chat_data.py` to recreate the dialogue fine-tuning files
from [OpenAssistant OASST1](https://huggingface.co/datasets/OpenAssistant/oasst1).
OASST1 is a human-generated, assistant-style conversation dataset distributed
under the Apache-2.0 license. The preparation script keeps short English
prompt/response pairs whose characters are compatible with the base model.

Downloaded corpora are intentionally excluded from Git. The scripts,
configuration, source links, and attribution remain committed so every data
file can be recreated.
