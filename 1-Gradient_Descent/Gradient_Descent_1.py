weight = [-5, 0, 5, 9, 10, 11, 15]
target = 10

for i in range(len(weight)):
    loss = (weight[i] - target) ** 2

    print("Initial weight:", weight[i])
    print("Initial loss:", loss)
    print("")