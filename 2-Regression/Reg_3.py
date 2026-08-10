# THIS ONE IS AI GENERATED NOT BY HAND BY ETHAN!!!

# Multiple linear regression using shared functions for every feature.
#
# Each row is one house: [size, bedrooms, constant bias input].
# The final input is always 1, which lets bias use the same multiplication
# and gradient formulas as every other weight.
features = [
    [1, 1, 1],
    [2, 1, 1],
    [1, 2, 1],
    [3, 1, 1],
    [2, 3, 1],
    [4, 2, 1],
]
prices = [6, 8, 9, 10, 14, 15]

# These should learn [2, 3, 1]. The last weight is the bias.
weight_names = ["Size weight", "Bedroom weight", "Bias"]
weights = [0, 0, 0]
learning_rate = 0.04


def predict(weights, inputs):
    prediction = 0
    for i in range(len(weights)):
        prediction += weights[i] * inputs[i]
    return prediction


def calculate_loss(weights, features, targets):
    total_squared_error = 0

    for i in range(len(features)):
        prediction = predict(weights, features[i])
        error = prediction - targets[i]
        total_squared_error += error ** 2

    return total_squared_error / len(features)


def calculate_gradients(weights, features, targets):
    gradients = [0] * len(weights)

    for i in range(len(features)):
        prediction = predict(weights, features[i])
        error = prediction - targets[i]

        # Every weight uses the same formula. Only its input is different.
        for j in range(len(weights)):
            gradients[j] += 2 * error * features[i][j]

    for j in range(len(gradients)):
        gradients[j] /= len(features)

    return gradients


for iteration in range(500):
    loss = calculate_loss(weights, features, prices)
    gradients = calculate_gradients(weights, features, prices)

    # Every weight uses the same gradient-descent update.
    for i in range(len(weights)):
        weights[i] -= learning_rate * gradients[i]

    if iteration == 0 or (iteration + 1) % 100 == 0:
        print(f"Iteration {iteration + 1}")
        print("Loss:", loss)
        for name, value in zip(weight_names, weights):
            print(f"{name}:", value)
        print("")


print("Final predictions:")
for i in range(len(features)):
    prediction = predict(weights, features[i])
    print(
        "Size:", features[i][0],
        "Bedrooms:", features[i][1],
        "Prediction:", round(prediction, 3),
        "Actual:", prices[i],
    )
