import random

text = "hello hello"
character_count = {}
for character in text:
    if character in character_count:
        character_count[character] += 1
    else:
        character_count[character] = 1
print(character_count)

character_probabilities = {}
for character, count in character_count.items():
    character_probabilities[character] = count / len(text)
print(character_probabilities)

characters = list(character_probabilities.keys())
probabilities = list(character_probabilities.values())

generated_text = ""

for i in range(100):
    selected_list = random.choices(
        characters,
        weights=probabilities,
        k=1
    )

    selected_character = selected_list[0]
    generated_text += selected_character

print("Generated text:")
print(generated_text)