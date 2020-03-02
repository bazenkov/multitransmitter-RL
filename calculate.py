import numpy as np


def calculate(c_neurons, c_ecs, c_iterations):

    # set start parameters
    c_time = np.array([0, 0])
    c_potentials = np.array([[n.u] * 2 for n in c_neurons])
    c_activations = np.array([[int(n.u >= n.u_th)] * 2 for n in c_neurons])
    c_ecs[0].calc_con(c_neurons, c_activations)
    tact_dur = np.array([0])
    u_rates = np.array([[n.u_rate(c_ecs, c_time, c_potentials[i])] * 2 for i, n in enumerate(c_neurons)])
    rhythm = None

    # start algorithm
    for i in range(c_iterations):

        # calculate MP changing rates due to ecs transmitter concentrations
        new_u_rates = np.array([[n.u_rate(c_ecs, c_time, c_potentials[i])] for i, n in enumerate(c_neurons)])
        u_rates = np.concatenate((u_rates, new_u_rates), axis=1)

        # calculate tact duration
        tact_dur = np.append(tact_dur, min([n.residual_time(new_u_rates[i]) for i, n in enumerate(c_neurons)]))

        # time of the next tact start
        c_time = np.append(c_time, c_time[-1] + tact_dur[-1])

        # update potentials and activation states
        new_potentials = np.array(
            [[n.update_potential(c_time, new_u_rates[i][0], c_potentials[i][-1])] for i, n in
             enumerate(c_neurons)]).reshape((len(c_neurons), 1))
        new_activations = np.array([[int(n.u >= n.u_th)] for n in c_neurons])
        c_potentials = np.concatenate((c_potentials, new_potentials), axis=1)
        c_activations = np.concatenate((c_activations, new_activations), axis=1)

        # calculate new transmitter concentrations
        for e in c_ecs:
            e.calc_con(c_neurons, c_activations)

    return c_neurons, c_ecs, c_time, c_potentials, c_activations, u_rates, tact_dur, rhythm
