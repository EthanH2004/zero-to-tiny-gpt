import numpy as np


# Model settings
text = (
    "the little cat sat by the window. "
    "the little dog slept beside the fire. "
    "the cat chased the mouse around the house. "
    "the mouse found cheese under the table. "
    "the dog followed the cat into the garden. "
    "the sun went down and the animals went to sleep. "
)

context_length = 5
embedding_size = 4
hidden_size = 32
learning_rate = 0.05
training_steps = 15000
generation_length = 300
generation_seed = None


# 1. Create the vocabulary and token IDs.
vocabulary = sorted(set(text))
vocabulary_size = len(vocabulary)

character_to_id = {}
id_to_character = {}

for token_id, character in enumerate(vocabulary):
    character_to_id[character] = token_id
    id_to_character[token_id] = character

print("\n=== 1. VOCABULARY ===")
print("Vocabulary:", vocabulary)
print("Character to ID:", character_to_id)


# 2. Encode the text as token IDs.
encoded_text = []

for character in text:
    token_id = character_to_id[character]
    encoded_text.append(token_id)

print("\n=== 2. ENCODED TEXT ===")
print(encoded_text)


# 3. Create context-target training examples.
contexts = []
targets = []

print("\n=== 3. CONTEXTS AND TARGETS ===")

for i in range(len(encoded_text) - context_length):
    context = encoded_text[i:i + context_length]
    target = encoded_text[i + context_length]

    contexts.append(context)
    targets.append(target)

    print(context, "->", target)


# 4. Convert the training examples into NumPy arrays.
context_matrix = np.array(contexts)
target_array = np.array(targets)

number_of_examples = context_matrix.shape[0]
mlp_input_size = context_length * embedding_size
row_numbers = np.arange(number_of_examples)

print("\n=== 4. TRAINING MATRICES ===")
print("Context matrix shape:", context_matrix.shape)
print("Target array shape:", target_array.shape)


# 5. Initialize all trainable parameters once.
np.random.seed(generation_seed)

embedding_table = np.random.randn(
    vocabulary_size,
    embedding_size
)

hidden_weights = np.random.randn(
    mlp_input_size,
    hidden_size
) * 0.1

hidden_bias = np.zeros(hidden_size)

output_weights = np.random.randn(
    hidden_size,
    vocabulary_size
) * 0.1

output_bias = np.zeros(vocabulary_size)

print("\n=== 5. TRAINABLE PARAMETERS ===")
print("Embedding table shape:", embedding_table.shape)
print("Hidden weight shape:", hidden_weights.shape)
print("Hidden bias shape:", hidden_bias.shape)
print("Output weight shape:", output_weights.shape)
print("Output bias shape:", output_bias.shape)


# 6. Train the MLP using forward passes, backpropagation, and gradient descent.
print("\n=== 6. TRAINING ===")

for training_step in range(training_steps + 1):
    # Forward pass
    embedded_contexts = embedding_table[context_matrix]

    flattened_contexts = embedded_contexts.reshape(
        number_of_examples,
        mlp_input_size
    )

    hidden_raw_values = (
        flattened_contexts @ hidden_weights
        + hidden_bias
    )

    hidden_activations = np.tanh(hidden_raw_values)

    logits = (
        hidden_activations @ output_weights
        + output_bias
    )

    stable_logits = logits - logits.max(
        axis=1,
        keepdims=True
    )

    exponentials = np.exp(stable_logits)
    probabilities = exponentials / exponentials.sum(
        axis=1,
        keepdims=True
    )

    # Cross-entropy loss
    correct_probabilities = probabilities[
        row_numbers,
        target_array
    ]

    loss = -np.log(correct_probabilities).mean()

    # Backpropagate through the output layer.
    logit_gradients = probabilities.copy()
    logit_gradients[row_numbers, target_array] -= 1
    logit_gradients /= number_of_examples

    output_weight_gradients = (
        hidden_activations.T @ logit_gradients
    )

    output_bias_gradients = logit_gradients.sum(axis=0)

    hidden_activation_gradients = (
        logit_gradients @ output_weights.T
    )

    # Backpropagate through tanh and the hidden layer.
    hidden_raw_gradients = (
        hidden_activation_gradients
        * (1 - hidden_activations ** 2)
    )

    hidden_weight_gradients = (
        flattened_contexts.T @ hidden_raw_gradients
    )

    hidden_bias_gradients = hidden_raw_gradients.sum(axis=0)

    flattened_context_gradients = (
        hidden_raw_gradients @ hidden_weights.T
    )

    # Backpropagate into the embedding table.
    embedded_context_gradients = (
        flattened_context_gradients.reshape(
            number_of_examples,
            context_length,
            embedding_size
        )
    )

    embedding_gradients = np.zeros_like(embedding_table)

    np.add.at(
        embedding_gradients,
        context_matrix,
        embedded_context_gradients
    )

    # Update every trainable parameter with gradient descent.
    embedding_table = (
        embedding_table
        - learning_rate * embedding_gradients
    )

    hidden_weights = (
        hidden_weights
        - learning_rate * hidden_weight_gradients
    )

    hidden_bias = (
        hidden_bias
        - learning_rate * hidden_bias_gradients
    )

    output_weights = (
        output_weights
        - learning_rate * output_weight_gradients
    )

    output_bias = (
        output_bias
        - learning_rate * output_bias_gradients
    )

    if training_step % 500 == 0:
        print("Step:", training_step, "Loss:", loss)


# 7. Generate text using the trained MLP.
current_context = text[:context_length]
generated_text = current_context

for generation_step in range(generation_length):
    # Encode the current context.
    current_context_ids = []

    for character in current_context:
        token_id = character_to_id[character]
        current_context_ids.append(token_id)

    current_context_array = np.array(current_context_ids)

    # Run one forward pass through the trained model.
    current_embeddings = embedding_table[current_context_array]

    current_flattened = current_embeddings.reshape(
        1,
        mlp_input_size
    )

    current_hidden = np.tanh(
        current_flattened @ hidden_weights
        + hidden_bias
    )

    current_logits = (
        current_hidden @ output_weights
        + output_bias
    )

    stable_current_logits = (
        current_logits[0]
        - current_logits[0].max()
    )

    current_exponentials = np.exp(stable_current_logits)
    current_probabilities = (
        current_exponentials
        / current_exponentials.sum()
    )

    # Sample the next character and slide the context forward.
    next_token_id = np.random.choice(
        vocabulary_size,
        p=current_probabilities
    )

    next_character = id_to_character[next_token_id]
    generated_text += next_character

    current_context = (
        current_context[1:]
        + next_character
    )

print("\n=== 7. GENERATED TEXT ===")
print(generated_text)
