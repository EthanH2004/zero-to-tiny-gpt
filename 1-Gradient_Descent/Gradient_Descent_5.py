weight = 0
target = 10
learning_rate = .1

for i in range(100):
    loss = (weight - target) ** 2
    slope = 2 * (weight - target)
    print("Current weight:", weight)
    print("Current loss:", loss)
    print("Slope:", slope)
    weight = weight - learning_rate * slope
