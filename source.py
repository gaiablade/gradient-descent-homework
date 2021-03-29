import csv
import time
import matplotlib.pyplot as plt
import copy

def normalize(X: list):
    for i in range(1, len(X[0])):
        feat_max = X[0][i]
        feat_min = X[0][i]
        for row in X:
            feat_max = max(feat_max, row[i])
            feat_min = min(feat_min, row[i])
        for row in X:
            row[i] = (row[i] - feat_min) / (feat_max - feat_min)
    return X

def hx(thetas: list, X: list):
    sum = 0
    for i in range(len(thetas)):
        sum += thetas[i] * X[i]
    return sum

def summation_regress(index: int, X: list, Y: list, thetas: list):
    sum = 0
    for row in range(len(X)):
        sum += (hx(thetas, X[row]) - Y[row]) * X[row][index]
    return sum

def summation_cost(X: list, Y: list, thetas: list):
    sum = 0
    for row in range(len(X)):
        sum += (hx(thetas, X[row]) - Y[row]) ** 2
    return sum

def MSE(X: list, Y: list, thetas: list):
    mse = (1/len(X)) * summation_cost(X, Y, thetas)
    return mse

def part_1():
    def gradientDiscent(X: list, Y: list, thetas: list, alpha: float, epoch: int):
        m = len(X)
        costs = []
        for i in range(epoch):
            thetas_copy = copy.deepcopy(thetas)
            # Update theta values
            for j in range(len(X[0])):
                thetas[j] = thetas[j] - alpha * (1/m) * summation_regress(j, X, Y, thetas_copy)
            if i % 100 == 0:
                # Calculate mse
                mse = MSE(X, Y, thetas)
                print("MSE at {0}: {1}".format(i, mse))
                costs.append(mse)
        return thetas, costs

    NUM_FEATURES = 0
    X = []
    Y = []

    with open("data/winequality-red.csv") as file:
        reader = csv.reader(file)

        header = next(reader)
        NUM_FEATURES = len(header) # 12

        for row in reader:
            X.append([1])
            for i in range(NUM_FEATURES - 1):
                X[-1].append(float(row[i]))
            Y.append(float(row[-1]))
    
    X = normalize(X)
    thetas = [0.5] * NUM_FEATURES

    (thetas, costs) = gradientDiscent(X, Y, thetas, alpha=0.001, epoch=1000)

    mse = MSE(X, Y, thetas)
    costs.append(mse)

    # Plot the cost (mean-squared error) over time
    plt.plot([100 * i for i in range(11)], costs)
    plt.xlabel(f'Iterations')
    plt.ylabel(f'Mean-Squared Error (MSE)')
    plt.title(f'Cost over time of data/winequality-red.csv')
    plt.show()

    print()
    print(f'The MSE after 1000 iterations: {mse}')
    print(f'Benchmark: < 1.5 MSE')
    print(f'Theta Values:')
    for i in range(len(thetas)):
        print(f'θ{i}: {thetas[i]}')
    input(f'Press Enter to continue...')
    print()

def part_2():
    def polynomialDescent(X: list, Y: list, thetas: list, alpha: float, epoch: int, order: int):
        costs = []
        m = len(X)
        for i in range(epoch):
            for j in range(len(X[0])):
                thetas_copy = copy.deepcopy(thetas)
                thetas[j] = thetas[j] - alpha * 1/len(X) * summation_regress(j, X, Y, thetas_copy)
            if i % int(epoch / 10) == 0:
                # calculate mse
                mse = MSE(X, Y, thetas)
                print(f"MSE at {i} iterations: {mse}")
                costs.append(mse)
        return thetas, costs

    # Part 2: Polynomial Regression Using Basis Expansion
    for filename in ["data/synthetic-1.csv", "data/synthetic-2.csv"]:
        X_unnormalized = []
        Y = []
        with open(filename) as file:
            reader = csv.reader(file)
            for row in reader:
                X_unnormalized.append(float(row[0]))
                Y.append(float(row[-1]))
        for order in [2, 3, 5]:
            NUM_FEATURES = order + 1
            X = []
            for i in range(0, len(X_unnormalized)):
                X.append([1.0] * NUM_FEATURES)
                for j in range(0, order):
                    X[i][j + 1] = X_unnormalized[i] ** (j + 1)
            
            # Normalize X
            X = normalize(X)
            thetas = [1] * (order + 1)

            (thetas, costs) = polynomialDescent(X, Y, thetas, 0.05, 10000, order)
            mse = MSE(X, Y, thetas)
            costs.append(mse)

            # Do Predictions
            x = []
            y = []
            for j in range(0,len(X)):
                x.append(X[j][1])
                y.append(Y[j])
            
            # Create dictionary
            predictions = dict()
            # Key: X value, Value: Y value
            for j in range(0, len(X)):
                predictions[X[j][1]] = hx(thetas, X[j])
            # Sort keys by numerical order
            xx = sorted(predictions.keys())
            # Sort values in the same order as the keys
            yy = [predictions[key] for key in xx]

            plt.scatter(x, y)
            plt.plot(xx, yy, color="red")
            plt.title(f'{filename} with order {order}')
            plt.xlabel(f'X (normalized)')
            plt.ylabel(f'Y (predicted values')
            plt.show()

            print()
            print(f'The MSE of {filename} after 10000 iterations at order {order}: {mse}')
            print(f'Theta Values:')
            for i in range(len(thetas)):
                print(f'θ{i}: {thetas[i]}')
            input(f'Press Enter to continue...')
            print()

if __name__ == "__main__":
    print(f'Part 1: Linear Regression with Gradient Descent')
    part_1()

    print(f'Part 2: Polynomial Regression with Basis Expansion')
    part_2()