import numpy as np


def get_burst_limits(activation_sequence, time):
    """Finds time limits of each burst.

    Iterates through every neighboring pair of elements in binary activation sequence of a neuron and finds the time
    of each activity change (from 1 to 0 and inverse).

    :param list activation_sequence: Binary sequence of activation function values of a neuron.
    :param list time: Sequence of float numbers defining the time of an event occurrence.
    :return: List of two-element lists where the first and the second elements are the the onset and
        the offset times of the burst respectively.
    :rtype: list
    """

    burst_limits = []

    for i, (preceding, sequent) in enumerate(zip(activation_sequence[:-1], activation_sequence[1:])):

        if (preceding == 0 and sequent == 1) or (i == 0 and preceding == 1):
            burst_limits.append([time[i + 1]])

        elif preceding == 1 and sequent == 0:
            burst_limits[-1].append(time[i + 1])

    if burst_limits and len(burst_limits[-1]) == 1:
        burst_limits[-1].append(time[-1])

    return burst_limits


class Neuron:

    def __init__(self, values):

        keys = (
            'name', 'u_th', 'u_max', 'u_0', 'u_min', 'u_reb', 'v_00', 'v_01', 'v_10', 'v_11', 'v_reb', 'u', 'd', 'w')
        self.__dict__.update(zip(keys, values))

        self.u_rate_end = 0

    def u_rate(self, ecs, time, potentials):

        impact = np.sum(np.array(self.w) * ecs[0].cons)
        u_rate_reb = 0

        if self.v_01 >= 0:
            if abs(self.u - self.u_0) < 5e-06 and potentials[-2] >= self.u_0:
                self.u_rate_end = self.v_01
            elif abs(self.u - self.u_th) < 5e-06 and potentials[-2] < self.u_th - 5e-06 and self.u >= self.u_th:
                self.u_rate_end = self.v_11
            elif abs(self.u - self.u_max) < 5e-06 and potentials[-2] >= self.u_th - 5e-06:
                self.u_rate_end = self.v_10
            elif potentials[-2] >= self.u_th - 5e-06 and self.u_0 + 5e-06 < self.u < self.u_th:
                self.u_rate_end = self.v_00

        elif self.v_01 < 0:
            if abs(self.u - self.u_0) < 5e-06:
                if impact == 0:
                    self.u_rate_end = 0
                elif impact > 0:
                    self.u_rate_end = self.v_01
                elif impact < 0:
                    self.u_rate_end = (-1) * self.v_01
            elif self.u_0 - self.u > 5e-06:
                self.u_rate_end = (-1) * self.v_01
            elif self.u - self.u_0 > 5e-06 > potentials[-2] - self.u_0:
                self.u_rate_end = self.v_01
            elif abs(self.u - self.u_th) < 5e-06 and potentials[-2] < self.u_th - 5e-06 and self.u >= self.u_th:
                self.u_rate_end = self.v_11
            elif abs(self.u - self.u_max) < 5e-06 and potentials[-2] >= self.u_th - 5e-06:
                self.u_rate_end = self.v_10
            elif potentials[-2] >= self.u_th - 5e-06 and self.u_0 + 5e-06 < self.u < self.u_th:
                self.u_rate_end = self.v_00

        if abs(self.u - self.u_reb) < 5e-06:
            u_rate_reb = self.v_reb
        elif abs(self.u - self.u_th) < 5e-06 and u_rate_reb == self.v_reb:
            u_rate_reb = 0
        if time[-1] == 0:
            u_rate_reb = 0

        return impact + self.u_rate_end + u_rate_reb

    def residual_time(self, u_rate):

        residual_time = 9999

        if self.v_01 < 0 and self.v_10 < 0 and self.u < self.u_0 - 5e-06 and u_rate > 0:
            residual_time = (self.u_0 - self.u) / u_rate


        elif self.u_max >= self.u > (self.u_th + 5e-06) and u_rate < 0:
            residual_time = -(self.u - self.u_th + (2e-06)) / u_rate

        elif (self.u_max - 5e-06) > self.u >= self.u_th and u_rate > 0:
            residual_time = (self.u_max - self.u) / u_rate

        elif (self.u_0 + 5e-06 < self.u <= self.u_th) and u_rate < 0:
            residual_time = -(self.u - self.u_0) / u_rate

        elif self.u < self.u_th - 5e-06 and u_rate > 0:
            residual_time = (self.u_th - self.u) / u_rate

        elif self.u_reb >= self.u_min and (self.u_reb < self.u < self.u_th) and u_rate < 0:
            residual_time = -(self.u - self.u_reb) / u_rate

        return residual_time

    def update_potential(self, time, u_rate, u_last):
        
        flag = 0
        zero_time = 0
        
        if zero_time == 0 and flag == 'u_max':
            self.u = self.u_max
        elif zero_time == 0 and flag == 'u_min':
            self.u = self.u_0
        else:
            self.u = u_last + (time[-1] - time[-2]) * u_rate

        if self.u > self.u_max:
            self.u = self.u_max
        elif self.u < self.u_min:
            self.u = self.u_min
        if abs(self.u - self.u_th) < 5e-07:
            self.u = self.u_th

        if abs(self.u - self.u_reb) < 5e-07:
            self.u = self.u_reb

        return self.u

    def update_activation(self):

        return int(self.u > self.u_th or self.u == self.u_th and self.u_rate > 0)

