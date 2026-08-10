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

total_probability = sum(character_probabilities.values())
print("Total probability:", total_probability)