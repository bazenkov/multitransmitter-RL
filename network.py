import csv
import numpy as np
from datetime import datetime
import sys
import matplotlib.pyplot as plt
import Neuron as nrn
from ECS import ECS
from plots import show_plot
from save_data import save_data

TIME_EPS_FRAC = 1e-3

def strict_slicing(seq, i_start, i_end):
    if i_start >= len(seq):
        raise ValueError("Index out of bounds")
    else:
        return seq[i_start:i_end]
        
class InputNeuron(nrn.Neuron):

    fields = ["name" , "u_th", "u_max", "u_0", "u_min", "u_reb", "v_00", "v_01", "v_10", "v_11", "v_reb", "u", "d", "w", "w_inputs"]

    def __init__(self, name, u_th, u_max, u_0, u_min, u_reb, v_00, v_01, v_10, v_11, v_reb, u, d, w, w_inputs):
        """Creates a neuron with external imputs. 
        input - a list [weight_1, weight_2]
        """
        nrn.Neuron.__init__(self, name, u_th, u_max, u_0, u_min, u_reb, v_00, v_01, v_10, v_11, v_reb, u, d, w)
        self.w_inputs = w_inputs

    def u_rate(self, ecs, time, potentials, inputs):
        """
        ecs - list of ECS objects
        time - list of tacts time points. time[-1] - the time when this tact started
        potentials - list of membrane potentials of the neuron for each tact
        inputs - numpy array of inputs at the current tact
        """
        impact = np.sum(np.array(self.w) * ecs[0].cons)
        if inputs.size:
            impact += np.sum(np.array(self.w_inputs) * inputs)
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
    
    def to_list(self):
        return [self.__dict__[k] for k in self.fields]

    def from_list(self, attr_values, attr_names):
        i_attr = 0
        for attr in attr_names:
            if np.isscalar(self.__dict__[attr]):
                len_attr = 1
                self.__dict__[attr] = attr_values[i_attr]
            else:
                len_attr = len(self.__dict__[attr])
                self.__dict__[attr] = list(strict_slicing(attr_values, i_attr, i_attr + len_attr) )
            i_attr += len_attr
            

class Network:

    @staticmethod
    def from_list(params):
        pass

    def __init__(self, neurons):
        self.neurons = neurons
        self.ecs = ECS()
        self.reset()
    
    def reset(self):
        """Initialize network before simulation
        """
        self.time = np.array([0, 0])
        self.potentials = np.array([[n.u] * 2 for n in self.neurons])
        self.activations = np.array([[int(n.u >= n.u_th)] * 2 for n in self.neurons])
        self.ecs.calc_con(self.neurons, self.activations)
        self.u_rates = np.array([[n.u_rate([self.ecs], self.time, self.potentials[i], inputs = np.array([]))] * 2 for i, n in enumerate(self.neurons)])
        self.tact_dur = np.array([0])

    def step_old(self, inputs, step_duration):
        """Calculate the dynamics for the specified duration with the fixed external inputs
        neurons - list of InputNeuron objects
        ecs - list of ECS
        inputs - numpy array of the same shape as w_inputs array of each neuron
        step_duration - duration in model units, corresponding to threshold and rates of the neurons
        potentials - history of previous neuron potentials as [[u_0, u_1, u_2, u_3, ...], [...], ... ]
        Returns neurons, ecs, time, potentials, activations, u_rates
        """
        total_duration = 0.0
        while total_duration < step_duration*(1 - TIME_EPS_FRAC):
            # calculate MP changing rates due to ecs transmitter concentrations
            new_u_rates = np.array([[n.u_rate([self.ecs], self.time, self.potentials[i], inputs)] 
                                for i, n in enumerate(self.neurons)])
            self.u_rates = np.concatenate((self.u_rates, new_u_rates), axis=1)
            # calculate tact duration
            self.tact_dur = np.append(self.tact_dur, 
                    min([n.residual_time(new_u_rates[i]) for i, n in enumerate(self.neurons)] + [step_duration-total_duration]))
            self.time = np.append(self.time, self.time[-1] + self.tact_dur[-1])
            #next potentials
            new_potentials = np.array(
                [[n.update_potential(self.time, new_u_rates[i][0], self.potentials[i][-1])] for i, n in
                    enumerate(self.neurons)]).reshape((len(self.neurons), 1))
            new_activations = np.array([[int(n.u >= n.u_th)] for n in self.neurons])
            self.potentials = np.concatenate((self.potentials, new_potentials), axis=1)
            self.activations = np.concatenate((self.activations, new_activations), axis=1)
            self.ecs.calc_con(self.neurons, new_activations)
            total_duration += self.tact_dur[-1]
        return new_activations

    def step(self, inputs, step_duration, injection = None):
        """Calculate the dynamics for the specified duration with the fixed external inputs
        neurons - list of InputNeuron objects
        ecs - list of ECS
        inputs - numpy array of the same shape as w_inputs array of each neuron
        step_duration - duration in model units, corresponding to threshold and rates of the neurons
        potentials - history of previous neuron potentials as [[u_0, u_1, u_2, u_3, ...], [...], ... ]
        Returns neurons, ecs, time, potentials, activations, u_rates
        """
        total_duration = 0.0
        while total_duration < step_duration*(1 - TIME_EPS_FRAC):
            self.ecs.calc_con(self.neurons, self.activations, injection)
            # calculate MP changing rates due to ecs transmitter concentrations
            new_u_rates = np.array([[n.u_rate([self.ecs], self.time, self.potentials[i], inputs)] 
                                for i, n in enumerate(self.neurons)])
            self.u_rates = np.concatenate((self.u_rates, new_u_rates), axis=1)
            # calculate tact duration
            self.tact_dur = np.append(self.tact_dur, 
                    min([n.residual_time(new_u_rates[i]) for i, n in enumerate(self.neurons)] + [step_duration-total_duration]))
            self.time = np.append(self.time, self.time[-1] + self.tact_dur[-1])
            #next potentials
            new_potentials = np.array(
                [[n.update_potential(self.time, new_u_rates[i][0], self.potentials[i][-1])] for i, n in
                    enumerate(self.neurons)]).reshape((len(self.neurons), 1))
            new_activations = np.array([[int(n.u >= n.u_th)] for n in self.neurons])
            self.potentials = np.concatenate((self.potentials, new_potentials), axis=1)
            self.activations = np.concatenate((self.activations, new_activations), axis=1)
            total_duration += self.tact_dur[-1]
        return new_activations
    
    def to_list(self):
        pass

    def plot(self):
        #c_neurons, c_ecs, c_time, c_potentials, c_activations, u_rates, tact_dur, rhythm
        params = [self.neurons, [self.ecs], self.time, self.potentials, self.activations, self.u_rates, self.tact_dur, []]
        fig = show_plot(params)
        #plt.gca()
        
        for ax in fig.axes:
            #ax.set_yticklabels([np.min(self.potentials), np.max(self.potentials)])
            ax.set_yticklabels(ax.get_yticks())
            ax.set_xticks(np.round(self.time[1::200],2))
            ax.set_xticklabels(ax.get_xticks())
        #plt.show()
    def num_neurons(self):
        return len(self.neurons)

def load_network(file_name):
    """
    Loads parameters of the system from a csv file. File must be located in a root directory

    :param file_name: Name of the file with parameters
    :return: Network object, number of iterations
    """
    l_neurons = []
    with open(file_name, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=';', quotechar='|')
        data = np.array(list(reader))
        l_iterations = int(data[0][1])
        for line in data[2:]:
            #['name', 'u_th', 'u_max', 'u_0', 'u_min', 'u_reb', 'v_00', 'v_01', 'v_10', 'v_11', 'v_reb', 'u', 'd', 'w', 'w_inputs']
            values = [line[0]] + [eval(i) for i in line[1:]]
            l_neurons.append(InputNeuron(*values))
    return Network(l_neurons), l_iterations

def save_network(network, file_name=None):
    """
    Saves start parameters of the system to a csv file in a root directory.

    :param list s_neurons: The list of neurons
    :param int s_iterations: Number of iterations
    :param str file_name: Name of the file. If not specified, the name of the file will contain current time and date.
    :return: None
    """
    s_iterations = len(network.time)
    if not file_name:
        date = datetime.strftime(datetime.now(), "%H.%M.%S - %Y.%m.%d")
        name = date + '.csv'
    else:
        if file_name[-4:] != '.csv':
            name = file_name + '.csv'
        else:
            name = file_name
    with open(name, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=';',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)

        writer.writerow(['iterations', s_iterations])
        writer.writerow(InputNeuron.fields)
        for n in network.neurons:
            writer.writerow(n.to_list())

if __name__ == "__main__":
    file = sys.argv[1]#"saved_params/ton_inp.csv"
    net, n_iter = load_network(file)
    #inputs = np.array([1, 0, 3.5, -2])
    inputs = np.zeros(4)
    net.neurons[0].from_list([0.75, 2, -2], attr_names=['u_th', 'w'])
    injection = np.array([0, 0])
    print(net.neurons[0].u_th)
    print(net.neurons[0].w)
    step_duration = 0.02
    for i in range(n_iter):
        net.step(inputs, step_duration, injection)
    net.plot()
    plt.show()

    #save_network(net, "trained/test_save.csv")
    #print(net.potentials)
    #print(net.activations)
    #print(net.time)