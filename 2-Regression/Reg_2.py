x = [1, 2, 3, 4]
y = [3, 5, 7, 9]
weight = 0
bias = 0
learning_rate = 0.05


def calculate_loss(weight, bias, x, y):
    total_squared_error = 0
    for i in range(len(x)):
        prediction = weight * x[i] + bias
        squared_error = (prediction - y[i]) ** 2
        total_squared_error += squared_error

    average_squared_error = total_squared_error / len(x)
    return average_squared_error

def calculate_weight_slope(weight, bias, x, y):
    total_weight_slope = 0
    for i in range(len(x)):
        prediction = weight * x[i] + bias
        error = prediction - y[i]
        weight_slope = 2 * error * x[i]
        total_weight_slope += weight_slope
    return total_weight_slope / len(x)


def calculate_bias_slope(weight, bias, x, y):
    total_bias_slope = 0
    for i in range(len(x)):
        prediction = weight * x[i] + bias
        error = prediction - y[i]
        bias_slope = 2 * error
        total_bias_slope += bias_slope
    return total_bias_slope / len(x)


for iteration in range(10):
    print("Iteration:", iteration + 1)
    print("Initial weight:", weight)
    print("Initial bias:", bias)

    initial_loss = calculate_loss(weight, bias, x, y)
    print("Initial loss:", initial_loss)
    
    current_weight_slope = calculate_weight_slope(weight, bias, x, y)
    current_bias_slope = calculate_bias_slope(weight, bias, x, y)
    print("Weight slope:", current_weight_slope)
    print("Bias slope:", current_bias_slope)

    weight = weight - learning_rate * current_weight_slope
    bias = bias - learning_rate * current_bias_slope
    print("Updated weight:", weight)
    print("Updated bias:", bias)
    print("")

print("Final weight:", weight)
print("Final bias:", bias)
print("Final predictions:", [weight * xi + bias for xi in x])
