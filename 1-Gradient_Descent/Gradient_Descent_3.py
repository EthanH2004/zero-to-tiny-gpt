weight = 0
target = 10
small_step = [1, 0.1, 0.01, 0.001, 0.0001]

for i in range(len(small_step)):
    current_loss = (weight - target) ** 2
    nearby_weight = weight + small_step[i]
    nearby_loss = (nearby_weight - target) ** 2
    slope = (nearby_loss - current_loss) / small_step[i]

    print("weight:", weight)
    print("Slope:", slope)
    print("")
