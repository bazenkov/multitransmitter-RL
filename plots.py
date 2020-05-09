import matplotlib.pyplot as plt
from Neuron import get_burst_limits
from sys import argv
import json

#Usage
#python plots.py --plot file_1 file_2
#python plots.py --save file_image file_1 file_2


def show_plot(p_params):
    """
    p_params = [c_neurons, c_ecs, c_time, c_potentials, c_activations, u_rates, tact_dur, rhythm]
    """
    p_neurons = p_params[0]
    p_potentials = p_params[3]
    fig = plt.figure()
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
        ax_set[-1].set_yticks([p_neurons[pic].u_min - 0.01, p_neurons[pic].u_th, p_neurons[pic].u_max + 0.01])
        #ax_set[-1].set_yticks([p_neurons[pic].u_th], minor=True)
        ax_set[-1].tick_params(axis='y', which='minor', labelsize=14, labelbottom=1)

        ax_set[-1].set_yticklabels([])

        if pic == len(p_neurons) - 1:
            ax_set[-1].xaxis.set_label_coords(1.01, -0.025)
        elif pic == 0:
            ax_set[-1].xaxis.set_label_coords(-0.03, 1.3)
    return fig

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

STYLES = ['b-', 'g--', 'r-.']

def plot_history(R_history, title='', labels=None, size=None):
    fig = plt.figure(figsize=tuple(size), dpi=150)
    
    if is_sequence(R_history[0]):
        for i,data in enumerate(R_history): 
            plt.plot(data, STYLES[i], linewidth=2)
    else:
        plt.plot(R_history)
    plt.gca().set_ylabel("Reward", size=12)
    plt.gca().set_xlabel("Iterations", size=12)
    plt.gca().tick_params(which='major', labelsize=12, labelbottom=1)
    #plt.gcf().figsize = tuple(size)
    fig.set_tight_layout(True)
    if title:
        plt.gca().set_title(title)
    if labels:
        plt.gca().legend(labels, loc='lower right', fontsize=12)
    
    #plt.gca().tick_params(axis='x', which='major', labelsize=12, labelbottom=1)

def is_sequence(obj):
    try: 
        len(obj)
        return True
    except:
        return False

if __name__ == "__main__":
    plot_conf = argv[1]
    params = json.load(open(plot_conf))
    R_history = []
    for file in params['files']:
        R_history.append(json.load(open(file))['reward_history'])
    plot_history(R_history, title=params['title'], labels=params['labels'], size=params['figsize'])
    if 'print_file' in params:
        plt.savefig(params['print_file'], format='png') 
    plt.show() 
    
    
    #if argv[1]=="--plot":
    #    ind = 2
    #    R_history = []
    #elif argv[1]=="--save":
    #    file_print = argv[2]
    #    ind = 3
    #else:
    #    raise ValueError("Incorrect command-line arguments!")
    #for file in argv[ind:]:
    #    R_history.append(json.load(open(file))['reward_history'])
    #plot_history(R_history)
    #if argv[2]=="--save" :
    #    plt.savefig(file_print, format='png')
    #plt.show()        
    #if len(argv)>3:
    #    title = argv[3]
    
        
    