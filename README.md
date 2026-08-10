# Machine learning from scratch

I'm teaching myself machine learning by writing every script by hand — no AI-generated code — building up from gradient descent to my own tiny GPT-style language model. These are small scripts that each work a tiny bit, and that's the point: every one exists so I understand the idea inside it.

## The projects

| Folder | What I learned | Final version |
|---|---|---|
| `1-Gradient_Descent` | Loss, slopes, stepping downhill with a learning rate | `Gradient_Descent_5.py` |
| `2-Regression` | Training a weight and bias with mean squared error | `Reg_2.py` |
| `3-Character_Probabilities` | Character counts, probabilities, random sampling | `Character_2.py` |
| `4-Bigram_Language_Model` | Conditional probabilities, generating text one character at a time | `Bigram_3.py` |
| `5-Neural_Bigram` | Softmax, cross-entropy, hand-derived gradients, a trained language model | `Neural_Bigram_1.py` |

The earlier numbered scripts in each folder are my stepping stones — I kept them because the progression is the learning. One exception for honesty: `2-Regression/Reg_3.py` was AI-generated as an example of multi-variable regression for me to study; everything else is hand-written.

Next up: an MLP language model with embeddings, then self-attention, then the tiny GPT.

## Running anything

Every script is standalone:

```
python <folder>/<script>.py
```

Only `5-Neural_Bigram` needs numpy; the rest are pure Python.

MIT licensed — use it however you like.
