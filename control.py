import sys
import os
from datetime import datetime
import gym
from gym.wrappers.monitoring.video_recorder import VideoRecorder
import numpy as np
import random
import network as net
from algorithms import nes
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

    def set_inputs(self, network, all_inputs):
        n_neurons = network.num_neurons()
        assert self.n_inputs == all_inputs.size // n_neurons
        for i_n, n in enumerate(network.neurons):
            n.w_inputs = list(all_inputs.reshape(n_neurons, self.n_inputs)[i_n,:])

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
        EnvWrap.__init__(self, 'CartPole-v0')
    
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

SCALE = 50
ALPHA = 0.01

def train(env_w, network, n_iter, pop_size, n_episode, episode_duration, sigma, alpha, gamma = 0):
    def fun(w, params):
        env_w.set_inputs(params[1], w)
        #params = [env, network, n_episode, episode_duration]
        return control(params[0], params[1], params[2], params[3])
    w, pop, R_history, best_history = nes(fun,  np.random.randn(env_w.num_weights(network)), n_iter, pop_size, sigma, alpha, gamma,
             [env_w, network, n_episode, episode_duration])
    return w, pop, R_history, best_history


def random_search(env_w, network, n_sample, sigma = 1, n_episode = 2, episode_duration=100):
    reward = np.zeros(n_sample)
    policies = []
    for i_p in range(n_sample):
        all_inputs = RNG.normal(scale = sigma, size=env_w.num_weights(network))
        env_w.set_inputs(network, all_inputs)
        policies.append(all_inputs)
        reward[i_p] = control(env_w, network, n_episode, episode_duration)
    print("Best policy is:")
    best_ind = np.argmax(reward)
    print(policies[best_ind])
    print("Best reward is: " + str(reward[best_ind]))
    env_w.set_inputs(network, policies[best_ind]) 
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


def test_random_search(file):
    network, _ = net.load_network(file)
    #env_w = CartPole()
    env_w = Acrobot()
    print(f"Starting {env_w.name} environment")
    #print(env.action_space)
    #print(env.observation_space)
    n_sample = 50
    network, _, _ = random_search(env_w, network, n_sample, sigma = 50, n_episode = 2, episode_duration=300)
    print("Show the best policy")
    control(env_w, network, n_episode = 1, episode_duration = 200, show = True)
    env_w.env.close()
    network.plot()
    plt.show()
    folder = "trained"
    file_name = folder + "/" + f"random_{n_sample}_" + datetime.strftime(datetime.now(), "%H.%M.%S - %Y.%m.%d") + ".csv"
    net.save_network(network, file_name)

def test_nes(file):
    network, n_iter = net.load_network(file)
    #start control cycle
    #env_w = CartPole()
    env_w = Acrobot()
    print(f"Starting {env_w.name} environment")
    print(env_w.env.action_space)
    print(env_w.env.observation_space)
    n_population = 50
    n_iter = 40
    best_w, pop, R_history, best_history = train(env_w, network, n_iter, n_population, 
        n_episode = 2, episode_duration = 300, sigma = 50, alpha = 20, gamma = 0.5)
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

def show_network(file, record = False, video_path = None):
    network, n_iter = net.load_network(file)
    #start control cycle
    #env_w = CartPole()
    env_w = Acrobot()
    print(f"Starting {env_w.name} environment")
    print(env_w.env.action_space)
    print(env_w.env.observation_space)
    #R = control(env, network, n_episode = 1, episode_duration = 200, show = False)
    if record:
        vr = VideoRecorder(env_w.env, base_path = video_path)
        R = control(env_w, network, n_episode = 1, episode_duration = 200, show = True, recorder = vr)
        vr.close()
    else:
        R = control(env_w, network, n_episode = 1, episode_duration = 200, show = True)
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
    
    option = sys.argv[1]
    if option == "nes":
        file = sys.argv[2]#"saved_params/ton_inp.csv"
        test_nes(file)
    elif option == "rand":
        file = sys.argv[2]#"saved_params/ton_inp.csv"
        test_random_search(file)
    elif option == "show":
        file = sys.argv[2]#"saved_params/ton_inp.csv"
        if len(sys.argv) > 3:
            video_file = sys.argv[3]            
            show_network(file, record = True, video_path = video_file)
        else:
            show_network(file)
    elif option == "test":
        test_video()

    
    

    

    

    


