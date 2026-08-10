import numpy as np


text = "hello hello"
learning_rate = 1.0


# 1. Create the vocabulary and token IDs.
vocabulary = sorted(set(text))
vocabulary_size = len(vocabulary)

character_to_id = {}
id_to_character = {}

for token_id, character in enumerate(vocabulary):
    character_to_id[character] = token_id
    id_to_character[token_id] = character

print("\n=== 1. VOCABULARY AND TOKEN IDs ===")
print("Vocabulary:", vocabulary)
print("Vocabulary size:", vocabulary_size)
print("Character to ID:", character_to_id)
print("ID to character:", id_to_character)


# 2. Encode the text as IDs, then decode it back into text.
encoded_text = []

for character in text:
    token_id = character_to_id[character]
    encoded_text.append(token_id)

decoded_text = ""

for token_id in encoded_text:
    character = id_to_character[token_id]
    decoded_text += character

print("\n=== 2. ENCODE AND DECODE ===")
print("Original text:", text)
print("Encoded text:", encoded_text)
print("Decoded text:", decoded_text)


# 3. Pair each current token with the token that follows it.
inputs = []
targets = []

for i in range(len(encoded_text) - 1):
    current_token = encoded_text[i]
    next_token = encoded_text[i + 1]

    inputs.append(current_token)
    targets.append(next_token)

print("\n=== 3. INPUTS AND TARGETS ===")
print("Inputs:", inputs)
print("Targets:", targets)


# 4. Turn every input ID into a one-hot vector.
one_hot_inputs = []

for token_id in inputs:
    one_hot_vector = [0] * vocabulary_size
    one_hot_vector[token_id] = 1
    one_hot_inputs.append(one_hot_vector)

one_hot_matrix = np.array(one_hot_inputs, dtype=float)

print("\n=== 4. ONE-HOT INPUT MATRIX ===")
print("Rows (current characters):", list(text[:-1]))
print("Columns:", vocabulary)
print(one_hot_matrix)
print("Shape:", one_hot_matrix.shape)


# 5. Give every current/next character pair one trainable weight.
weights = np.zeros((vocabulary_size, vocabulary_size))

print("\n=== 5. WEIGHT MATRIX ===")
print("Rows (current characters):", vocabulary)
print("Columns (possible next characters):", vocabulary)
print(weights)
print("Shape:", weights.shape)


# 6. Train the model using gradient descent.
target_ids = np.array(targets)
row_numbers = np.arange(len(targets))
number_of_examples = len(inputs)

print("\n=== 6. TRAINING ===")

for training_step in range(5000):
    # Forward pass
    logits = one_hot_matrix @ weights

    exponentials = np.exp(logits)
    probabilities = exponentials / exponentials.sum(
        axis=1,
        keepdims=True
    )

    # Loss
    correct_probabilities = probabilities[row_numbers, target_ids]
    loss = -np.log(correct_probabilities).mean()

    # Gradients
    logit_gradients = probabilities.copy()
    logit_gradients[row_numbers, target_ids] -= 1
    logit_gradients /= number_of_examples

    weight_gradients = one_hot_matrix.T @ logit_gradients

    # Gradient-descent update
    weights = weights - learning_rate * weight_gradients

    if training_step % 100 == 0:
        print("Step:", training_step, "Loss:", loss)


print("\n=== 7. TRAINED WEIGHTS ===")
print("Rows (current characters):", vocabulary)
print("Columns (possible next characters):", vocabulary)
print(weights)