"""
    A simple TrajectoryListener that averages the total score and average entropy over the last x trajectories
        and puts these statistics in a live matplotlib graph.
"""
from utility.trajectory_listener import TrajectoryListener
import matplotlib.pyplot as plt
import numpy as np


class AvgScoreEntropyTrajectoryListener(TrajectoryListener):
    def __init__(self, n, task_ids, colours):
        """
        Initializes the listener
        :param task_ids: a list of task_ids to plot
        :param colours: the colours used to plot the given task_ids
        :param n: Number of averaged trajectories
            (keep in mind that it will only average over all available trajectories enough trajectories are available)
        """
        super().__init__()
        self.colours = colours
        self.task_ids = task_ids
        self.entropies = []             # Keeps a list of the n last entropy values
        self.scores = []                # Keeps a list of the n last scores
        self.avg_entropies = []         # Keeps a list of all the plotted average entropy values
        self.avg_scores = []            # Keeps a list of all the plotted average score values
        self.n = n
        self.plt1 = plt.subplot(1, 2, 1)
        self.plt2 = plt.subplot(1, 2, 2)

    def process_trajectory(self, trajectory):
        reward_sum = np.sum([r for _, _, r, _, _ in trajectory], axis=0)
        entropy_mean = np.mean([np.sum(-p * np.log2(p)) for _, _, _, p, _ in trajectory])
        self.entropies.append(entropy_mean)
        self.scores.append(reward_sum)

        if len(self.scores) == self.n:
            score_avg = np.mean(self.scores, axis=0)
            entropy_avg = np.mean(self.entropies)

            self.avg_scores.append(score_avg)
            self.avg_entropies.append(entropy_avg)

            avg_scores = np.array(self.avg_scores)

            self.plt1.clear()
            self.plt2.clear()
            for task_id, colour in zip(self.task_ids, self.colours):
                self.plt1.plot(avg_scores[:, task_id], color=colour)
            self.plt2.plot(self.avg_entropies)
            plt.pause(0.05)
            self.scores = []
            self.entropies = []
