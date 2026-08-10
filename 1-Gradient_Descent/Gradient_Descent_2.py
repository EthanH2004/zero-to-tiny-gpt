weight = 0
target = 10
small_step = 0.001

current_loss = (weight - target) ** 2

nearby_weight = weight + small_step
nearby_loss = (nearby_weight - target) ** 2

slope = (nearby_loss - current_loss) / small_step

print("Current weight:", weight)
print("Current loss:", current_loss)
print("Nearby weight:", nearby_weight)
print("Nearby loss:", nearby_loss)
print("Slope:", slope)