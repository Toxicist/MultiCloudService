import numpy as np
import matplotlib.pyplot as plt


def plot_window_reward(scores, filename, x=None, window=5):
    print("-------------------------Saving Picture-------------------------")
    N = len(scores)

    running_avg = np.empty(N)

    for t in range(N):
        running_avg[t] = np.mean(scores[max(0, t-window):(t+1)])

    if x is None:
        x = [i for i in range(N)]

    plt.ylabel('Score')
    plt.xlabel('Game')

    plt.plot(x, running_avg, "r", label="our")
    plt.legend()

    plt.savefig(filename)
    plt.close()


def plot_reward(reward, filename):
    print('-' * 20 + 'Savting Picture' + '-' * 20)
    N = len(reward)
    x = [i for i in range(N)]

    plt.ylabel('Reward')
    plt.xlabel('Episode')

    plt.plot(x, reward, "r", label="our")
    plt.legend()

    plt.savefig(filename)
    plt.close()

if __name__ == '__main__':
    plot_reward(1, 2)
