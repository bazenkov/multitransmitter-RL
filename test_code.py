import matplotlib.pyplot as plt

from Neuron import Neuron
from calculate import calculate
from ECS import ECS
from save_data import save_data, load_data
from plots import show_plot

neurons, ecs, iterations = load_data('saved_params/olp1.csv')
params = calculate(neurons, ecs, iterations)
show_plot(params)
plt.show()
