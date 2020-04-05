import sys
import os
from datetime import datetime
import gym
from gym.wrappers.monitoring.video_recorder import VideoRecorder
import numpy as np
import random
import network as net
from algorithms import nes
from algorithms import simple_ga
import matplotlib.pyplot as plt


"""Command-line options
python contro.py {show, train, test} network_filename 

"""

RNG = np.random.default_rng()



class EnvWrap:

    def __init__(self, name):
        self.name = name
        self.env = gym.make(self.name)
        self.n_inputs = np.prod(self.env.observation_space.shape)

    def num_weights(self, network):
        return self.n_inputs*network.num_neurons()

    def get_action(self, activation):
        raise NotImplementedError()

class CartPole(EnvWrap):
    LEFT_NEURON = 0
    RIGHT_NERON = 1
    LEFT = 0
    RIGHT = 1
    
    def __init__(self):
        EnvWrap.__init__(self, 'CartPole-v1')
    
    def get_action(self, activation):
        if activation[self.LEFT_NEURON]==1 and activation[self.RIGHT_NERON]==0:
            return self.LEFT
        if activation[self.LEFT_NEURON]==0 and activation[self.RIGHT_NERON]==1:
            return self.RIGHT
        return random.randrange(2)

    def set_inputs(self, network, all_inputs):
        n_neurons = network.num_neurons()
        assert self.n_inputs == all_inputs.size // n_neurons
        for i_n, n in enumerate(network.neurons):
            n.w_inputs = list(all_inputs.reshape(n_neurons, self.n_inputs)[i_n,:])

    def num_weights(self, network):
        return self.n_inputs*network.num_neurons()

    def time_step(self):
        return self.env.tau

class Acrobot(EnvWrap):
    LEFT_NEURON = 0
    RIGHT_NERON = 1
    LEFT = -1
    RIGHT = 1
    ZERO = 0
    
    def __init__(self):
        EnvWrap.__init__(self, 'Acrobot-v1')
    
    def get_action(self, activation):
        if activation[self.LEFT_NEURON]==1 and activation[self.RIGHT_NERON]==0:
            return self.LEFT
        if activation[self.LEFT_NEURON]==0 and activation[self.RIGHT_NERON]==1:
            return self.RIGHT
        elif activation[self.LEFT]==0 and activation[self.RIGHT]==0:
            return self.ZERO
        return self.env.action_space.sample()

    def time_step(self):
        return self.env.dt

def create_env(name):
    if name == "Acrobot":
        return Acrobot()
    elif name == "Cartpole":
        return CartPole()
    raise NotImplementedError()


def set_inputs(network, vector_attr, attr_names):
    n_neurons = network.num_neurons()
    assert vector_attr.size == vector_attr.shape[0]
    assert vector_attr.size % n_neurons == 0
    matr_attr = vector_attr.reshape(n_neurons, vector_attr.size // n_neurons)
    for i_n, n in enumerate(network.neurons):
        n.from_list(list(matr_attr[i_n,:]), attr_names)


SCALE = 50
ALPHA = 0.01

def fun(w, params):
    set_inputs(params[1], w, params[2])
    #params = [env, network, attr_names, n_episode, episode_duration]
    return control(params[0], params[1], params[3], params[4])

def train(env_w, network, attr_names, n_iter, pop_size, n_episode, episode_duration, sigma, alpha, gamma = 0):
    w, pop, R_history, best_history = nes(fun,  np.random.randn(env_w.num_weights(network)), n_iter, pop_size, sigma, alpha, gamma,
             [env_w, network, attr_names, n_episode, episode_duration])
    return w, pop, R_history, best_history

def attr_size(neuron, attr_names):
    s = 0
    for name in attr_names:
        if np.isscalar(neuron.__dict__[name]):
            s += 1
        else:
            s += len(neuron.__dict__[name])
    return s


def random_search(env_w, network, attr_names, n_sample, sigma = 1, n_episode = 2, episode_duration=100):
    param_size = attr_size(network.neurons[0], attr_names)*network.num_neurons()
    reward = np.zeros(n_sample)
    policies = []
    for i_p in range(n_sample):
        w = RNG.normal(scale = sigma, size=param_size)
        set_inputs(network, w, attr_names)
        policies.append(w)
        reward[i_p] = control(env_w, network, n_episode, episode_duration)
    print("Best policy is:")
    best_ind = np.argmax(reward)
    print(policies[best_ind])
    print("Best reward is: " + str(reward[best_ind]))
    set_inputs(network, policies[best_ind], attr_names) 
    return network, policies, reward

def control(env_w, network, n_episode=1, episode_duration=100, show = False, recorder = None):
    cum_reward = 0.0
    for _ in range(n_episode):
        inputs = env_w.env.reset()
        network.reset()
        for _ in range(episode_duration):
            if show:
                if recorder:
                    recorder.capture_frame()
                else:
                    env_w.env.render()
            activation = network.step(inputs, step_duration =  env_w.time_step())
            inputs, reward, done, _ =  env_w.env.step( env_w.get_action(activation))
            cum_reward += reward
            if done:
                break
   # print("Total reward is: " + str(cum_reward))
    return cum_reward/n_episode


def test_random_search(file, attr_names):
    network, _ = net.load_network(file)
    env_w = CartPole()
    #env_w = Acrobot()
    print(f"Starting {env_w.name} environment")
    #print(env.action_space)
    #print(env.observation_space)
    n_sample = 300
    network, _, _ = random_search(env_w, network, attr_names, n_sample, sigma = 50, n_episode = 5, episode_duration=500)
    print("Show the best policy")
    control(env_w, network, n_episode = 1, episode_duration = 500, show = True)
    env_w.env.close()
    network.plot()
    plt.show()
    folder = "trained"
    file_name = folder + "/" + f"random_{n_sample}_" + datetime.strftime(datetime.now(), "%H.%M.%S - %Y.%m.%d") + ".csv"
    net.save_network(network, file_name)

def test_nes(file, attr_names):
    network, n_iter = net.load_network(file)
    #start control cycle
    env_w = CartPole()
    #env_w = Acrobot()
    print(f"Starting {env_w.name} environment")
    print(env_w.env.action_space)
    print(env_w.env.observation_space)
    n_population = 50
    n_iter = 40
    best_w, pop, R_history, best_history = train(env_w, network, attr_names,  n_iter, n_population, 
        n_episode = 3, episode_duration = 300, sigma = 50, alpha = 10, gamma = 0.0)
    print("Show the best policy")
    env_w.set_inputs(network, best_w)
    #control(env, network, n_episode = 1, episode_duration = 100, show = True)
    #network.plot()
    env_w.env.close()
    plt.plot(R_history)
    plt.show()
    folder = "trained"
    file_name = folder + "/" + f"nes_" + datetime.strftime(datetime.now(), "%H.%M.%S - %Y.%m.%d") + ".csv"
    net.save_network(network, file_name)

def train_ga(file, attr_names):
    network, n_iter = net.load_network(file)
    param_size = attr_size(network.neurons[0], attr_names)*network.num_neurons()
    #start control cycle
    env_w = CartPole()
    #env_w = Acrobot()
    print(f"Starting {env_w.name} environment")
    print(env_w.env.action_space)
    print(env_w.env.observation_space)
    n_population = 50
    n_iter = 10
    elite_frac = 0.1
    w_0 = np.random.randn(param_size)
    n_episode = 1
    episode_duration = 300
    best_w, pop, R_history, best_history = simple_ga(fun, w_0, 
            n_iter = 10, pop_size = 50, elite_frac = 0.1, sigma = 50,
            params = [env_w, network, attr_names, n_episode, episode_duration])
    print("Show the best policy")
    set_inputs(network, best_w, attr_names)
    env_w.env.close()
    plt.plot(R_history)
    folder = "trained/" + env_w.name
    file_name = folder + "/" + f"ga_" + datetime.strftime(datetime.now(), "%H.%M.%S - %Y.%m.%d") + ".csv"
    net.save_network(network, file_name)
    return file_name

def show_network(file, record = False, video_path = None):
    network, n_iter = net.load_network(file)
    #start control cycle
    env_w = CartPole()
    #env_w = Acrobot()
    print(f"Starting {env_w.name} environment")
    print(env_w.env.action_space)
    print(env_w.env.observation_space)
    #R = control(env, network, n_episode = 1, episode_duration = 200, show = False)
    if record:
        vr = VideoRecorder(env_w.env, base_path = video_path)
        R = control(env_w, network, n_episode = 1, episode_duration = 500, show = True, recorder = vr)
        vr.close()
    else:
        R = control(env_w, network, n_episode = 1, episode_duration = 500, show = True)
    env_w.env.close()
    network.plot()
    plt.show()
    print(f"Average reward R={R}")

def test_video():
    env = gym.make("CartPole-v1")
    rec = VideoRecorder(env, base_path = "fig/test")
    env.reset()
    rec.capture_frame()
    rec.close()
    assert not rec.empty
    assert not rec.broken
    assert os.path.exists(rec.path)
    print(f"Video saved to {rec.path}")
    f = open(rec.path)
    assert os.fstat(f.fileno()).st_size > 100

if __name__ == "__main__":
    #create environment
    attr_names = ['w_inputs']
    option = sys.argv[1]
    if len(sys.argv)>2:
        file = sys.argv[2]#"saved_params/ton_inp.csv"
    if option == 'nes' or option == 'rand' or option == 'ga':
        if len(sys.argv) > 3:
            attr_names = eval(sys.argv[3])
    if option == 'nes':
        test_nes(file, attr_names)
    elif option == 'rand':
        test_random_search(file, attr_names)
    elif option == 'ga':
        file_name = train_ga(file, attr_names)
        show_network(file_name)
    elif option == 'show':
        if len(sys.argv) > 3:
            video_file = sys.argv[3]            
            show_network(file, record = True, video_path = video_file)
        else:
            show_network(file)
    elif option == 'test':
        test_video()

    
    

    

    

    


