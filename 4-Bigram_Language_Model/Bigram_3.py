import random

text = "hello hello"
transition_counts = {}
transition_probabilities = {}

for i in range(len(text) - 1):
    current_character = text[i]
    next_character = text[i + 1]

    if current_character not in transition_counts:
        transition_counts[current_character] = {}

    if next_character in transition_counts[current_character]:
        transition_counts[current_character][next_character] += 1
    else:
        transition_counts[current_character][next_character] = 1
    
for current_character, next_counts in transition_counts.items():
    total = sum(next_counts.values())
    transition_probabilities[current_character] = {}

    for next_character, count in next_counts.items():
        probability = count / total
        transition_probabilities[current_character][next_character] = probability

print(transition_counts)
print(transition_probabilities)

current_character = "h"
generated_text = current_character

for i in range(100):
    if current_character not in transition_probabilities:
        break
    next_probabilities = transition_probabilities[current_character]

    characters = list(next_probabilities.keys())
    probabilities = list(next_probabilities.values())

    selected_list = random.choices(
        characters,
        weights=probabilities,
        k=1
    )

    selected_character = selected_list[0]
    generated_text += selected_character

    current_character = selected_character

print("Generated text:")
print(generated_text)