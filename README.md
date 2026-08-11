# Zero to tiny GPT

I wanted to actually understand how language models work instead of just watching videos about them. So I started with a gradient descent script small enough to fit on an index card and kept building until I had a little GPT chatbot trained on my Mac.

The numbered folders are the practice units, in order. I wrote those scripts myself. I used AI as a tutor along the way (it reviewed my code and nudged me with things like "now write a loop that counts character pairs"), but the code in units 1 through 7 is mine, typed by me, dead ends included. Each folder keeps every version I wrote, so you can watch gradient descent go from "try seven weights in a list and print the loss" to the real update rule.

1. Gradient descent. Follow the slope downhill until the loss hits zero.
2. Regression. Fit y = 2x + 1 with a hand-written training loop.
3. Character probabilities. Count letters, sample new text from the counts.
4. Bigram model. Predict the next character from the current one. My first language model.
5. Neural bigram. The same model, but trained. Softmax, cross-entropy, and the gradients derived by hand.
6. MLP. Five characters of context, embeddings, a hidden layer, backprop written out in numpy.
7. Self-attention. Queries, keys, values, and a causal mask, printed one step at a time.

Tiny_GPT and Tiny_GPT_v2 are the finish line, and I'll be honest about them: I built these with AI help, in PyTorch. By that point the goal wasn't to type every line myself. I'd already hand-written embeddings, backprop, and attention, and I wanted to put the pieces together into something that runs. v1 is a character-level model trained on TinyStories. v2 is the do-over after I learned what actually mattered: a 2,048-piece BPE tokenizer, about 16 million parameters, 8 layers, trained overnight on FineWeb-Edu and OpenAssistant data from Hugging Face.

Does it work? A tiny bit. It chats, it mostly stays on topic, and I know what every file in it is for. That was the point.

Units 1 through 4 are plain Python. Units 5 through 7 need numpy. The two Tiny GPT folders need PyTorch; setup instructions are in Tiny_GPT/README.md.

MIT license. Use whatever you want.
