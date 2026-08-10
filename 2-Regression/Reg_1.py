x = [1, 2, 3, 4]
y = [2, 4, 6, 8]

weight = 0
total_squared_error = 0

for i in range(len(x)):
    prediction = weight * x[i]
    squared_error = (prediction - y[i]) ** 2
    total_squared_error += squared_error
    print("Input:", x[i])
    print("Prediction:", prediction)
    print("Actual:", y[i])
    print("Squared Error:", squared_error)
    print(" ")

average_squared_error = total_squared_error / len(x)
print("Total Squared Error:", total_squared_error)
print("Average Squared Error:", average_squared_error)
