import matplotlib.pyplot as plt
from Neuron import get_burst_limits


def show_plot(p_params):
    p_neurons = p_params[0]
    p_potentials = p_params[3]
    print(p_potentials)
    plt.figure()
    names = [n.name for n in p_neurons]
    names.reverse()
    ax_set = []
    for pic, dat in enumerate(p_neurons):

        ax_set.append(plt.subplot(len(p_neurons), 1, pic + 1))
        ax_set[-1].plot(len(p_neurons), 1, pic + 1)
        ax_set[-1].plot(p_params[2], p_potentials[pic], linewidth=2, linestyle="-",
                        label=p_neurons[pic].name)

        plt.legend(loc='upper left')

        plt.ylim(p_neurons[pic].u_min - 0.01, p_neurons[pic].u_max + 0.01)
        ax_set[-1].spines['right'].set_color('none')
        ax_set[-1].spines['top'].set_color('none')
        ax_set[-1].spines['bottom'].set_position(('data', p_neurons[pic].u_th))
        ax_set[-1].get_xticks()
        ax_set[-1].set_xticks([])
        ax_set[-1].set_yticks([p_neurons[pic].u_th])
        ax_set[-1].set_xticklabels([])
        ax_set[-1].set_yticks([p_neurons[pic].u_min - 0.01, p_neurons[pic].u_max + 0.01])
        ax_set[-1].set_yticks([p_neurons[pic].u_th], minor=True)
        ax_set[-1].tick_params(axis='y', which='minor', labelsize=14, labelbottom=1)

        ax_set[-1].set_yticklabels([])

        if pic == len(p_neurons) - 1:
            ax_set[-1].xaxis.set_label_coords(1.01, -0.025)
        elif pic == 0:
            ax_set[-1].xaxis.set_label_coords(-0.03, 1.3)


def population_plot(p_params):
    p_neurons = p_params[0]
    fig, ax = plt.subplots(figsize=(8, len(p_neurons) * 0.3))

    names = [n.name for n in p_neurons]
    names.reverse()
    for i, neuron in enumerate(p_neurons):
        bursts = get_burst_limits(p_params[4][i], p_params[2])
        for pack in bursts:

            plt.plot(pack, [len(p_neurons) - i] * 2, color='k')
            ax.set_yticks(list(range(len(p_neurons) + 1)))
            ax.set_yticklabels([''] + names)
            ax.spines['right'].set_color('none')
            ax.spines['top'].set_color('none')
            ax.spines['bottom'].set_color('none')
            ax.spines['left'].set_color('none')
            plt.xlim(0, max(p_params[2]))