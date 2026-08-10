text = "hello hello"
bigram_counts = {}
for i in range(len(text) - 1):
    bigram = text[i:i+2]
    if bigram in bigram_counts:
        bigram_counts[bigram] += 1
    else:
        bigram_counts[bigram] = 1
print(bigram_counts)
