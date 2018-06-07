"""
    This file defines a standard TrajectoryListener that can be registered with the SAC-X agents.
    The TrajectoryListener can be used to plot scores, entropy and other useful data from the received trajectories.

    The TrajectoryListener continuously receives trajectories and calls process_trajectory when a trajectory comes in.
"""
import queue
import sys
from multiprocessing import Process, Queue


class TrajectoryListener(Process):
    def __init__(self):
        super().__init__()
        self.trajectory_queue = Queue()

    def run(self):
        while True:
            trajectory = self.trajectory_queue.get()
            if self.trajectory_queue.qsize() > 10000:
                print("Listener trajectory queue exceeded 10000 entries, flushing queue!", file=sys.stderr)
                self._flush_queue()
            self.process_trajectory(trajectory)

    def process_trajectory(self, trajectory):
        """
        Receives a trajectory and processes it. This function can be implemented by subclasses.
        :param trajectory: a list of (state, action, reward vector, policy distribution, task_id)
        """
        pass

    def put_trajectory(self, trajectory):
        self.trajectory_queue.put(trajectory)

    def _flush_queue(self):
        while True:
            try:
                self.trajectory_queue.get(block=False)
            except queue.Empty:
                break