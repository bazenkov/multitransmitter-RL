import numpy as np


class ECS:

    def __init__(self):
        self.name = 'ECS '

    def calc_con(self, neurons, activations):

        self.cons = np.array([0] * len(neurons[0].d))

        for i, n in enumerate(neurons):
            self.cons = np.add(self.cons, np.array(n.d) * activations[:, -1][i])
