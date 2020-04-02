from datetime import datetime
import csv
import numpy as np

from Neuron import Neuron
from ECS import ECS


def save_data(s_neurons, s_iterations, file_name=None):
    """
    Saves start parameters of the system to a csv file in a root directory.

    :param list s_neurons: The list of neurons
    :param int s_iterations: Number of iterations
    :param str file_name: Name of the file. If not specified, the name of the file will contain current time and date.
    :return: None
    """
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
        writer.writerow(list(s_neurons[0].__dict__.keys())[:14])
        for n in s_neurons:
            writer.writerow(list(n.__dict__.values())[:14])


def load_data(file_name):
    """
    Loads parameters of the system from a csv file. File must be located in a root directory

    :param file_name: Name of the file with parameters
    :return: Tuple with:
                    List of neurons,
                    List with a single ECS object,
                    Number of iterations (int).
    """
    l_neurons = []
    with open(file_name, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=';', quotechar='|')
        data = np.array(list(reader))
        l_iterations = int(data[0][1])
        for line in data[2:]:
            #['name', 'u_th', 'u_max', 'u_0', 'u_min', 'u_reb', 'v_00', 'v_01', 'v_10', 'v_11', 'v_reb', 'u', 'd', 'w']
            values = [line[0]] + [eval(i) for i in line[1:]]
            l_neurons.append(Neuron(*values))

    l_ecs = [ECS()]

    return l_neurons, l_ecs, l_iterations