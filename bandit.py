import numpy as np
import random as rnd

class Restaurant:
    def __init__(self, happiness, distribution):
        self.happiness = happiness
        self.distribution = distribution

# Restaurant values
restaurants = [Restaurant(10,0.2), Restaurant(8,2), Restaurant(9,5), Restaurant(2,0)]

numberOfAttempts = 50000

threshold = 6.1  # How good does the restaurant need to be?

restaurantCount = len(restaurants)
X = np.zeros((numberOfAttempts, restaurantCount))
reward = np.zeros(restaurantCount)
punishment = np.zeros(restaurantCount)

# Prime 'happiness' values for all attempts in all restaurants
for i in range(numberOfAttempts):
    for j in range(restaurantCount):
        distribution = restaurants[j].distribution
        distribution_update = rnd.uniform(-distribution, distribution)

        happiness = restaurants[j].happiness
        happiness += distribution_update

        X[i][j] = happiness

# Visit a restaurant many times
for i in range(numberOfAttempts):
    selected = 0
    maxRandom = 0
    for j in range(restaurantCount):
        randomBeta = np.random.beta(reward[j] + 1, punishment[j] + 1)

        if randomBeta > maxRandom:
            maxRandom = randomBeta
            selected = j

    if X[i][selected] >= threshold:
        reward[selected] += 1
    else:
        punishment[selected] += 1


selected = reward + punishment
for i in range(restaurantCount):
    print('Restaurant: ' + str(i+1) + ' was selected ' + str(selected[i]) + ' times')
print('Best restaurant: ' + str(np.argmax(selected) + 1))