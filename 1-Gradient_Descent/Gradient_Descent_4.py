weight = 0
target = 10
learning_rate = 0.1

loss = (weight - target) ** 2

slope = 2 * (weight - target)

print("Current weight:", weight)
print("Current loss:", loss)
print("Slope:", slope)

weight = weight - learning_rate * slope
print("Updated weight:", weight)

updated_loss = (weight - target) ** 2
print("Updated loss:", updated_loss)