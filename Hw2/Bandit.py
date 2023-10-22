"""
  Run this file at first, in order to see what is it printng. Instead of the print() use the respective log level
"""

"""
Python script is for simulating and comparing the performance of two algorithms: 
Epsilon-Greedy and Thompson Sampling. We are going to perform 20000 trials and 
we have initial bandit rewards of 1,2,3,4.
"""
############################### LOGGER
from abc import ABC, abstractmethod
from logs import *
import matplotlib.pyplot as plt
import numpy as np
import random
import csv

# define the number of trials and Bandit Rewards
Bandit_Rewards = [1, 2, 3, 4]
NumberOfTrials = 20000
EPS = 0.1

logging.basicConfig
logger = logging.getLogger("MAB Application")


# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

ch.setFormatter(CustomFormatter())

logger.addHandler(ch)



class Bandit(ABC):
    ##==== DO NOT REMOVE ANYTHING FROM THIS CLASS ====##

    @abstractmethod
    def __init__(self, p):
        # Stores the probability of success for the bandit arm
        self.p = p
        # tracks the number of times the bandit arm has been pulled
        self.N = 0
        # stores the running average of rewards obtained from the bandit arm
        self.mean = 0

    # returns a string that includes the class name (Bandit) and the p value for the bandit arm
    def __repr__(self):
        return f"{self.__class__.__name__}({self.p})"

    # defines exploit-explore dilemma
    @abstractmethod
    def pull(self):
        pass

    @abstractmethod
    def update(self, x):
        pass

    # conducts a series of trials, pulling the bandit arm and recording rewards
    # returns cumulative rewards
    @abstractmethod
    def experiment(self, N):
        pass

    # logs and stores the results of a bandit algorithm's performance
    def report(self):
        # store the data in csv
        with open('Rewards.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([self.__class__.__name__, self.mean])

        # print the average reward
        logger.info(f"Average reward for {self.__class__.__name__}: {self.mean:.4f}")

#--------------------------------------#

class Visualization():

    def plot1(self, epsilon_greedy_cumulative_rewards, thompson_sampling_cumulative_rewards):
        # Visualize the performance of each bandit in linear scale
        plt.figure(figsize=(10, 5))
        plt.plot(epsilon_greedy_cumulative_rewards, label="Epsilon-Greedy")
        plt.plot(thompson_sampling_cumulative_rewards, label="Thompson Sampling")
        plt.xlabel("Number of Trials")
        plt.ylabel("Cumulative Rewards")
        plt.legend()
        plt.title("Bandit Algorithm Performance (Cumulative Rewards - Linear Scale)")
        plt.grid(True)
        plt.show()

        # Visualize the performance of each bandit in log scale
        plt.figure(figsize=(10, 5))
        plt.plot(np.log(epsilon_greedy_cumulative_rewards), label="Epsilon-Greedy")
        plt.plot(np.log(thompson_sampling_cumulative_rewards), label="Thompson Sampling")
        plt.xlabel("Number of Trials")
        plt.ylabel("Cumulative Rewards (Log Scale)")
        plt.legend()
        plt.title("Bandit Algorithm Performance (Cumulative Rewards - Log Scale)")
        plt.grid(True)
        plt.show()

    def plot2(self, epsilon_greedy_cumulative_rewards, thompson_sampling_cumulative_rewards, epsilon_greedy_cumulative_regrets, thompson_sampling_cumulative_regrets):
        # Compare E-greedy and Thompson Sampling cumulative rewards
        plt.figure(figsize=(10, 5))
        plt.plot(epsilon_greedy_cumulative_rewards, label="Epsilon-Greedy Rewards")
        plt.plot(thompson_sampling_cumulative_rewards, label="Thompson Sampling Rewards")
        plt.xlabel("Number of Trials")
        plt.ylabel("Cumulative Rewards")
        plt.legend()
        plt.title("Comparison of Cumulative Rewards")
        plt.grid(True)
        plt.show()

        # Compare E-greedy and Thompson Sampling cumulative regrets
        plt.figure(figsize=(10, 5))
        plt.plot(epsilon_greedy_cumulative_regrets, label="Epsilon-Greedy Regrets")
        plt.plot(thompson_sampling_cumulative_regrets, label="Thompson Sampling Regrets")
        plt.xlabel("Number of Trials")
        plt.ylabel("Cumulative Regrets")
        plt.legend()
        plt.title("Comparison of Cumulative Regrets")
        plt.grid(True)
        plt.show()


#--------------------------------------#

class EpsilonGreedy(Bandit):
    def __init__(self, p, epsilon):
        super().__init__(p)
        self.epsilon = epsilon

    def pull(self):
        if np.random.random() < self.epsilon:
            return np.random.random() < self.p
        else:
            return self.mean < np.random.random()

    def update(self, x):
        self.N += 1
        self.mean = (1 - 1.0 / self.N) * self.mean + 1.0 / self.N * x
        self.epsilon /= (self.N + 1)

    def experiment(self, N):
        rewards = []
        cumulative_rewards = []
        for i in range(N):
            x = self.pull()
            rewards.append(x)
            cumulative_rewards.append(sum(rewards))
            self.update(x)
        return cumulative_rewards


class ThompsonSampling(Bandit):
    def __init__(self, p, alpha, beta):
        super().__init__(p)
        self.alpha = alpha
        self.beta = beta

    def pull(self):
        return np.random.beta(self.alpha, self.beta) < self.p

    def update(self, x):
        self.N += 1
        self.alpha += x
        self.beta += 1 - x
        self.mean = self.alpha / (self.alpha + self.beta)

    def experiment(self, NUMBER_OF_TRIALS):
        rewards = []
        cumulative_rewards = []
        for i in range(NUMBER_OF_TRIALS):
            x = self.pull()
            rewards.append(x)
            cumulative_rewards.append(sum(rewards))
            self.update(x)
        return cumulative_rewards


def comparison(N):
    bandit_return = [1, 2, 3, 4]
    epsilons = [0.1, 0.2, 0.3]

    for bandit_reward in bandit_return:
        for eps in epsilons:
            # Epsilon-Greedy
            epsilon_greedy_bandit = EpsilonGreedy(bandit_reward, eps)
            epsilon_greedy_cumulative_rewards = epsilon_greedy_bandit.experiment(N)

            # Thompson Sampling
            alpha = 1
            beta = 1
            thompson_sampling_bandit = ThompsonSampling(bandit_reward, alpha, beta)
            thompson_sampling_cumulative_rewards = thompson_sampling_bandit.experiment(N)

    # Create an instance of the Visualization class
    visualization = Visualization()

    # Plot the cumulative rewards for comparison
    visualization.plot1(epsilon_greedy_cumulative_rewards, thompson_sampling_cumulative_rewards)

    # Now, call the plot2 function to compare cumulative rewards and regrets
    epsilon_greedy_bandit = EpsilonGreedy(1, EPS)
    thompson_sampling_bandit = ThompsonSampling(1, 1, 1)

    epsilon_greedy_cumulative_rewards = epsilon_greedy_bandit.experiment(N)
    thompson_sampling_cumulative_rewards = thompson_sampling_bandit.experiment(N)

    epsilon_greedy_cumulative_regrets = np.cumsum(Bandit_Rewards[-1] - np.array(epsilon_greedy_cumulative_rewards))
    thompson_sampling_cumulative_regrets = np.cumsum(
        Bandit_Rewards[-1] - np.array(thompson_sampling_cumulative_rewards))

    visualization.plot2(epsilon_greedy_cumulative_rewards, thompson_sampling_cumulative_rewards,
                        epsilon_greedy_cumulative_regrets, thompson_sampling_cumulative_regrets)


if __name__ == '__main__':
    N = NumberOfTrials
    p = Bandit_Rewards

    # Call the comparison function to run experiments and compare the algorithms
    comparison(N)

    # logger.debug("debug message")
    # logger.info("info message")
    # logger.warning("warning message")
    # logger.error("error message")
    # logger.critical("critical message")
