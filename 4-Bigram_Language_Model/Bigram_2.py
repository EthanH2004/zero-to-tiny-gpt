text = "hello hello"
transition_counts = {}

for i in range(len(text) - 1):
    current_character = text[i]
    next_character = text[i + 1]

    if current_character not in transition_counts:
        transition_counts[current_character] = {}

    if next_character in transition_counts[current_character]:
        transition_counts[current_character][next_character] += 1
    else:
        transition_counts[current_character][next_character] = 1

print(transition_counts)