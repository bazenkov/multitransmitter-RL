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

def _plain_arr(obj):
    if isinstance(obj, np.ndarray):
        return list(obj.reshape(1, obj.size))
    else:
        return obj

class InputNeuron(nrn.Neuron):

    fields = ["name" , "u_th", "u_max", "u_0", "u_min", "u_reb", "v_00", "v_01", "v_10", "v_11", "v_reb", "u", "d", "w", "w_inputs", "i_ecs"] 

    def _shape_d(self):
        raise NotImplementedError()
        n_tr = len(self.w)
        n_ecs = len(self.d) // n_tr
        self.d = np.array(self.d).reshape(n_ecs, n_tr)

    def __init__(self, name, u_th, u_max, u_0, u_min, u_reb, v_00, v_01, v_10, v_11, v_reb, u, d, w, w_inputs, i_ecs = 0):
        """Creates a neuron with external imputs. 
        input - a list [weight_1, weight_2]
        """
        nrn.Neuron.__init__(self, name, u_th, u_max, u_0, u_min, u_reb, v_00, v_01, v_10, v_11, v_reb, u, d, w)
        self.w_inputs = w_inputs
        self.i_ecs = i_ecs
        #self._shape_d()
        

    def u_rate(self, ecs, time, potentials, inputs):
        """
        ecs - the ECS object
        time - list of tacts time points. time[-1] - the time when this tact started
        potentials - list of membrane potentials of the neuron for each tact
        inputs - numpy array of inputs at the current tact
        """
        impact = np.sum(np.array(self.w) * ecs.cons)
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
        return [_plain_arr(self.__dict__[k]) for k in self.fields]

    def from_list_old(self, attr_values, attr_names):
        '''attr_names = ["w_inputs", "v_01"]
        attr_values = [0.5, 2, -1, 4, 2.35]
        '''
        i_attr = 0
        for attr in attr_names:
            if np.isscalar(self.__dict__[attr]):
                len_attr = 1
                self.__dict__[attr] = attr_values[i_attr]
            else:
                len_attr = len(self.__dict__[attr])
                self.__dict__[attr] = list(strict_slicing(attr_values, i_attr, i_attr + len_attr) )
            i_attr += len_attr

    def from_list(self, attr_values, d_attr_names):
        '''d_attr_names = {"w_inputs":[0, 2], "v_01":True}
        attr_values = [0.5, -1, 2.35]
        '''
        i_attr = 0
        for attr in d_attr_names:
            if np.isscalar(self.__dict__[attr]):
                len_attr = 1
                self.__dict__[attr] = attr_values[i_attr]
            else:
                len_attr = len(d_attr_names[attr])
                for i, ind in enumerate(d_attr_names[attr]):
                    self.__dict__[attr][ind] = attr_values[ i_attr + i]
            i_attr += len_attr

    def attr_size(self, d_attr_names):
        '''attr_names is a dict like:
        {"d": [0], "w_inputs":[0,2], "v_01":true }
        '''
        s = 0
        for name in d_attr_names.keys():
            if np.isscalar(self.__dict__[name]):
                s += 1
            else:
                s += len(d_attr_names[name])
        return s


class Network:

    @staticmethod
    def from_list(params):
        pass

    def __init__(self, neurons):
        self.neurons = neurons
        self.ecs = [ECS() for _ in range(self.num_ecs())]
        self.reset()
    
    def _update_ecs_single(self, injection = None):
        self.ecs[0].calc_con(self.neurons, self.activations, injection)

    def _get_d(self, i_neuron, i_ecs):
        i_start = i_ecs*self.num_transmitters()
        i_end = i_start + self.num_transmitters()
        return np.array(self.neurons[i_neuron].d[i_start:i_end])

    def _update_ecs_many(self, injection = None):
        for i_e, e in enumerate(self.ecs):
            e.cons = np.zeros(self.num_transmitters())
            for i_n, _ in enumerate(self.neurons):
                np.add(e.cons, self._get_d(i_n, i_e) * self.activations[:, -1][i_n], out = e.cons)
            if not (injection is None):
                e.cons = np.add(e.cons, injection[i_e, :])

    def _update_ecs(self, injection = None):
        if self.num_ecs() == 1:
            self._update_ecs_single(injection)
        else:
            self._update_ecs_many(injection)
            

    def reset(self):
        """Initialize network before simulation
        """
        self.time = np.array([0, 0])
        self.potentials = np.array([[n.u] * 2 for n in self.neurons])
        self.activations = np.array([[int(n.u >= n.u_th)] * 2 for n in self.neurons])
        self._update_ecs()
        self.u_rates = np.array([[n.u_rate(self.ecs[n.i_ecs], self.time, self.potentials[i], inputs = np.array([]))] * 2 for i, n in enumerate(self.neurons)])
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
            self._update_ecs()
            # calculate MP changing rates due to ecs transmitter concentrations
            new_u_rates = np.array([[n.u_rate(self.ecs[n.i_ecs], self.time, self.potentials[i], inputs)] 
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
        params = [self.neurons, self.ecs, self.time, self.potentials, self.activations, self.u_rates, self.tact_dur, []]
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

    def num_transmitters(self):
        return len(self.neurons[0].w)
    
    def num_ecs(self):
        return max([n.i_ecs for n in self.neurons])+1

    def set_attr_values(self, vector_attr, l_attr_names):
        '''l_attr_names is a list of dicts like
        [ {"d": [0], "w_inputs":[0,1,2,3], "v_01":true }, {"d": [1], "w_inputs":[0,1,2,3]}],

        vector_attr is a list
        '''
        #n_neurons = network.num_neurons()
        #assert vector_attr.size % n_neurons == 0
        #matr_attr = vector_attr.reshape(n_neurons, vector_attr.size // n_neurons)
        i_start = 0
        for i_n, n in enumerate(self.neurons):
            i_end = i_start + n.attr_size(l_attr_names[i_n])
            n.from_list(vector_attr[i_start:i_end], l_attr_names[i_n])
            i_start = i_end
    
    def total_attr_size(self, l_attr_names):
        return sum( [ n.attr_size(l_attr_names[i]) for i,n in enumerate(self.neurons) ])

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
            #['name', 'u_th', 'u_max', 'u_0', 'u_min', 'u_reb', 'v_00', 'v_01', 'v_10', 'v_11', 'v_reb', 'u', 'd', 'w', 'w_inputs', 'i_ecs']
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
    inputs = np.array([0, 0, 0, 0])
    #inputs = np.zeros(4)
    l_attr_names = [{'u_th':True, 'd':[1, 3]}, {'v_01':True, 'w':[0]}, {}, {}]
    net.neurons[0].from_list([0.75, 1.5, 6], d_attr_names={'u_th':True, 'd':[0, 2]})
    net.set_attr_values([0.55, 10, 15, 3.2, -4], l_attr_names)
    assert net.total_attr_size(l_attr_names) == 5
    injection = np.array([0, 0])
    print(net.neurons[0].u_th)
    print(net.neurons[0].w)
    step_duration = 1
    for i in range(n_iter):
        net.step(inputs, step_duration, injection)
    net.plot()
    plt.show()

    #save_network(net, "trained/test_save.csv")
    #print(net.potentials)
    #print(net.activations)
    #print(net.time)