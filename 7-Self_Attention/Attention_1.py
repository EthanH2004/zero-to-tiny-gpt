import numpy as np


# Model settings
text = "the cat"
embedding_size = 3
attention_size = 3
random_seed = 42

np.random.seed(random_seed)
np.set_printoptions(precision=3, suppress=True)


# 1. Create the vocabulary and token IDs.
vocabulary = sorted(set(text))
vocabulary_size = len(vocabulary)

character_to_id = {}
id_to_character = {}

for token_id, character in enumerate(vocabulary):
    character_to_id[character] = token_id
    id_to_character[token_id] = character

print("\n=== 1. VOCABULARY ===")
print("Text:", text)
print("Vocabulary:", vocabulary)
print("Character to ID:", character_to_id)


# 2. Encode the text.
token_ids = []

for character in text:
    token_ids.append(character_to_id[character])

token_ids = np.array(token_ids)
sequence_length = len(token_ids)

print("\n=== 2. ENCODED TEXT ===")
print("Token IDs:", token_ids)


# 3. Give each token a learned-number representation.
embedding_table = np.random.randn(vocabulary_size, embedding_size)
token_embeddings = embedding_table[token_ids]

print("\n=== 3. TOKEN EMBEDDINGS ===")
print(token_embeddings)
print("Shape:", token_embeddings.shape)


# 4. Add information about where each token appears.
position_ids = np.arange(sequence_length)
position_embedding_table = np.random.randn(sequence_length, embedding_size)
position_embeddings = position_embedding_table[position_ids]
input_embeddings = token_embeddings + position_embeddings

print("\n=== 4. TOKEN + POSITION EMBEDDINGS ===")
print("Position IDs:", position_ids)
print(input_embeddings)
print("Shape:", input_embeddings.shape)


# 5. Create one query, key, and value vector for every position.
query_weights = np.random.randn(embedding_size, attention_size)
key_weights = np.random.randn(embedding_size, attention_size)
value_weights = np.random.randn(embedding_size, attention_size)

queries = input_embeddings @ query_weights
keys = input_embeddings @ key_weights
values = input_embeddings @ value_weights

print("\n=== 5. QUERIES, KEYS, AND VALUES ===")
print("Queries:")
print(queries)
print("\nKeys:")
print(keys)
print("\nValues:")
print(values)
print("\nShape of each:", queries.shape)


# 6. Compare every query with every key.
attention_scores = queries @ keys.T
scaled_scores = attention_scores / np.sqrt(attention_size)

print("\n=== 6. SCALED ATTENTION SCORES ===")
print("Rows and columns:", list(text))
print(scaled_scores)
print("Shape:", scaled_scores.shape)


# 7. Prevent positions from looking at future positions.
causal_mask = np.triu(
    np.ones((sequence_length, sequence_length), dtype=bool),
    k=1,
)

masked_scores = scaled_scores.copy()
masked_scores[causal_mask] = -np.inf

print("\n=== 7. CAUSAL MASK ===")
print(causal_mask.astype(int))
print("\nMasked scores:")
print(masked_scores)


# 8. Convert the scores into attention percentages.
stable_scores = masked_scores - masked_scores.max(axis=1, keepdims=True)
exponentials = np.exp(stable_scores)
attention_weights = exponentials / exponentials.sum(axis=1, keepdims=True)

print("\n=== 8. ATTENTION WEIGHTS ===")
print(attention_weights)
print("Row totals:", attention_weights.sum(axis=1))


# 9. Mix the value vectors using the attention percentages.
attention_output = attention_weights @ values

print("\n=== 9. FIRST ATTENTION HEAD OUTPUT ===")
print(attention_output)
print("Shape:", attention_output.shape)


# 10. Repeat the same attention process with a second set of weights.
query_weights_2 = np.random.randn(embedding_size, attention_size)
key_weights_2 = np.random.randn(embedding_size, attention_size)
value_weights_2 = np.random.randn(embedding_size, attention_size)

queries_2 = input_embeddings @ query_weights_2
keys_2 = input_embeddings @ key_weights_2
values_2 = input_embeddings @ value_weights_2

attention_scores_2 = (queries_2 @ keys_2.T) / np.sqrt(attention_size)

masked_scores_2 = attention_scores_2.copy()
masked_scores_2[causal_mask] = -np.inf

stable_scores_2 = masked_scores_2 - masked_scores_2.max(axis=1, keepdims=True)
exponentials_2 = np.exp(stable_scores_2)
attention_weights_2 = exponentials_2 / exponentials_2.sum(
    axis=1,
    keepdims=True,
)

attention_output_2 = attention_weights_2 @ values_2


# 11. Join the two attention-head outputs side by side.
multi_head_output = np.concatenate(
    (attention_output, attention_output_2),
    axis=1,
)

print("\n=== 10. MULTI-HEAD ATTENTION OUTPUT ===")
print("Head 1 shape:", attention_output.shape)
print("Head 2 shape:", attention_output_2.shape)
print("Combined shape:", multi_head_output.shape)
print(multi_head_output)
