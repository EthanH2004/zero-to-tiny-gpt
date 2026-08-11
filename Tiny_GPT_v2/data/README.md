# Tiny GPT V2 data

`prepare_base_data.py` creates a 100-million-character educational-English
sample from Hugging Face FineWeb-Edu (`sample-10BT`, ODC-By license).

`prepare_chat_data.py` creates complete English conversation paths from the
OpenAssistant OASST1 ready conversation trees (Apache-2.0 license).

The generated text files are excluded from Git and can be recreated by the
overnight pipeline.

