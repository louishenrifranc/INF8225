from collections import defaultdict
from termcolor import cprint
from model import Model
from state import *
import numpy as np
import os.path
import pickle
import random
import shutil
import time
import uuid


class QTable:
    def __init__(self, M, K=50):
        self.replay_memory = list()
        self.best_accuracy = {}
        self._restore()
        self.M = M
        self.K = K

        self.epsilon = 1.0
        self.alpha = 0.01
        self.gamma = 1



        self.nb_to_keeps = 10

    def _restore(self):
        """
        Restore QTable and memory replay or create a new QTable
        :return:
        """
        if not os.path.exists("data"):
            os.makedirs("data")
        if not os.path.exists("model"):
            os.makedirs("model")
        self._q_table_path = os.path.join("data", "q_table.p")
        self._mem_replay_path = os.path.join("data", "mem_replay.p")
        self._best_accuracy_path = os.path.join("data", "best_accuracy.p")
        if os.path.exists(self._q_table_path) and os.path.exists(self._mem_replay_path):
            cprint("[*] Restoring QTable and memory replay", color="yellow")
            self.q_table = pickle.load(open(self._q_table_path, "rb"))
            self.replay_memory = pickle.load(open(self._mem_replay_path, "rb"))
            self.best_accuracy = pickle.load(open(self._best_accuracy_path, "rb"))
            self.episode = len(self.replay_memory)
            print(len(self.best_accuracy))
        else:
            self.q_table = defaultdict(list)
            self._buildQTable()
            self.episode = 0

    def _buildQTable(self):
        """
        Build QTable
        :return:
        """
        cprint("[*] Building QTable", color="yellow")

        # Build convolution state
        layer_depths = np.arange(1, 4)
        receptive_field_sizes = np.array([1, 3])
        strides = [1]
        receptive_fields = np.array([32, 64, 128])
        representation_sizes = [(9, 33), (5, 9), (1, 4)]
        all_convolution_states = []
        for layer_depth in layer_depths:
            for receptive_field_size in receptive_field_sizes:
                for stride in strides:
                    for receptive_field in receptive_fields:
                        for representation_size in representation_sizes:
                            all_convolution_states.append(ConvolutionState(layer_depth,
                                                                           receptive_field_size,
                                                                           stride,
                                                                           receptive_field,
                                                                           representation_size))

        # Build pooling state
        layer_depths = np.arange(1, 4)
        pair_receptieve_field_strides = [(3, 2), (2, 2), (2, 1)]
        representation_sizes = [(9, 33), (5, 9), (1, 4)]
        all_pooling_states = []
        for layer_depth in layer_depths:
            for pair_receptieve_field_stride in pair_receptieve_field_strides:
                for representation_size in representation_sizes:
                    all_pooling_states.append(PoolingState(layer_depth,
                                                           pair_receptieve_field_stride,
                                                           representation_size))
        # Build fully_connected
        layer_depths = np.arange(1, 4)
        consecutive_fc_layers = np.arange(1, 3)
        nb_neurons = np.array([128, 64, 32])
        all_fc_layers = []
        for layer_depth in layer_depths:
            for consecutive_fc_layer in consecutive_fc_layers:
                for nb_neuron in nb_neurons:
                    if not (layer_depth == 1 and consecutive_fc_layer == 2):
                        all_fc_layers.append(FullyConnectedState(layer_depth,
                                                                 consecutive_fc_layer,
                                                                 nb_neuron))
        """
        C TRANSITION
        ------------
        Rules
        -----
            C -> FC: (nb_channels <= 2 biggest nb_channels)
            C -> C: (nb_channels not equal biggest nb_channels && same_representation bin)
            C -> T: No explicit rules
            c -> P: (nb_channels not equal biggest nb_channels &&
                                           if pool stride == 1, then same bin
                                           if pool stride == 2, then pool bin is the smallest bin after conv bin)
        """
        # Transition between C to FC
        for convolution_state in all_convolution_states:
            for fc_state in all_fc_layers:
                if fc_state.consecutive_fc_layers == 1 \
                        and fc_state.layer_depth == convolution_state.layer_depth + 1 \
                        and convolution_state.layer_depth != layer_depths[-1] \
                        and convolution_state.representation_size in representation_sizes[-2:]:
                    self.q_table[convolution_state].append([fc_state, 0.5])
                    # print(convolution_state, "->", fc_state)

        # Transition between C to C
        for convolution_state in all_convolution_states:
            for next_convolution_state in all_convolution_states:
                if convolution_state.layer_depth + 1 == next_convolution_state.layer_depth \
                        and convolution_state.representation_size in representation_sizes[:-1] \
                        and convolution_state.layer_depth != layer_depths[-1] \
                        and convolution_state.representation_size == next_convolution_state.representation_size:
                    self.q_table[convolution_state].append([next_convolution_state, 0.5])
                    # print(convolution_state, '->', next_convolution_state)

        # Transition between C and T
        for convolution_state in all_convolution_states:
            if convolution_state.layer_depth == layer_depths[-1]:
                prob = 1
            else:
                prob = 0.5
            self.q_table[convolution_state].append([TerminateState(convolution_state), prob])

        # Transition between C and P
        # Heuristic for simplicity, always change to smaller representation_size
        for convolution_state in all_convolution_states:
            for pooling_state in all_pooling_states:
                if convolution_state.representation_size != representation_sizes[-1] \
                        and convolution_state.layer_depth != layer_depths[-1] \
                        and convolution_state.layer_depth + 1 == pooling_state.layer_depth:
                    if pooling_state.pair_receptive_field_size_strides[1] == 1 \
                            and convolution_state.representation_size[0] == pooling_state.representation_size[0]:
                        self.q_table[convolution_state].append([pooling_state, 0.5])
                        # print(convolution_state, '=>', pooling_state)
                    elif pooling_state.pair_receptive_field_size_strides[1] != 1 \
                            and convolution_state.representation_size[0] - 1 == pooling_state.representation_size[1]:
                        self.q_table[convolution_state].append([pooling_state, 0.5])
                        # print(convolution_state, '->', pooling_state)

        """
        FC transition
        -------------
        Rules
        -----
            FC -> FC (nb_previous_fc == 1 && nb_hidden should decrease)
            FC -> T: No explicit rules
        """

        # Transition between FC and FC
        for fc_state in all_fc_layers:
            for next_fc_state in all_fc_layers:
                if fc_state.consecutive_fc_layers < next_fc_state.consecutive_fc_layers and \
                                fc_state.nb_hidden > next_fc_state.nb_hidden and \
                                        fc_state.layer_depth + 1 == next_fc_state.layer_depth:
                    self.q_table[fc_state].append([next_fc_state, 0.5])
                    # print(fc_state, "->", next_fc_state)

        # Transition between FC and T
        for fc_state in all_fc_layers:
            if fc_state.consecutive_fc_layers == consecutive_fc_layers[-1] or \
                            fc_state.layer_depth == layer_depths[-1]:
                prob = 1
            else:
                prob = 0.5
            self.q_table[fc_state].append([TerminateState(fc_state), prob])

        """
        P Transition
        ------------
        Rules
        -----
            P -> C: (same bin representation && bin_representation not in (1, 4))
            P -> FC: (nb_channels <= biggest 2 channel size)
            P -> T: No explicit rules
        """
        # Transition between P and C
        for pooling_state in all_pooling_states:
            for convolution_state in all_convolution_states:
                if pooling_state.layer_depth + 1 == convolution_state.layer_depth \
                        and pooling_state.layer_depth != layer_depths[-1] \
                        and pooling_state.representation_size[0] == convolution_state.representation_size[0] \
                        and convolution_state.representation_size[0] != 1:
                    self.q_table[pooling_state].append([convolution_state, 0.5])
                    # print(pooling_state, '->', convolution_state)

        # Transition between P and FC:
        for pooling_state in all_pooling_states:
            for fc_state in all_fc_layers:
                if pooling_state.layer_depth + 1 == fc_state.layer_depth \
                        and pooling_state.layer_depth != layer_depths[-1] \
                        and pooling_state.representation_size in representation_sizes[-2:] \
                        and fc_state.consecutive_fc_layers == 1:
                    self.q_table[pooling_state].append([fc_state, 0.5])
                    # print(pooling_state, '->', fc_state)

        for pooling_state in all_pooling_states:
            if pooling_state.layer_depth == layer_depths[-1]:
                prob = 1
            else:
                prob = 0.5
            if pooling_state not in self.q_table:
                self.q_table[pooling_state] = []
            self.q_table[pooling_state].append([TerminateState(pooling_state), prob])
            # print(np.mean([len(self.q_table.p[key]) for key, _ in self.q_table.p.items()]))
        cprint(" Built", color="green", end="\r")

    def _sample_new_network(self, epsilon):
        """
        Sample a new network architecture
        :param epsilon: Integer
            Exploration ratio
        :return: List of States
            A new network architecture
        """
        cprint("[*] Sample new architecture", "green")
        founded = False
        while not founded:
            network_architecture = list()
            network_architecture.append(
                random.choice([state for state in self.q_table.keys() if state.layer_depth == 1]))
            try:
                while type(network_architecture[-1]) != TerminateState:
                    rand = random.random()
                    if rand > epsilon:
                        next_state = self._find_best_action(network_architecture[-1])
                    else:
                        next_state = self._find_random_action(network_architecture[-1])
                    network_architecture.append(next_state)
                founded = True
            except:
                cprint("[*] Finding a new architecture", color="red")
                founded = False
        cprint("[*] New architecture of length {}: {}".format(len(network_architecture), network_architecture),
               color="green")
        return network_architecture

    def _find_best_action(self, state):
        """
        Return the best state to go following the optimal policy
        :param state:
        :return:
        """
        best_state = (None, 0)
        random.shuffle(self.q_table[state])
        for next_state in self.q_table[state]:
            if next_state[1] > best_state[1]:
                best_state = next_state
        return best_state[0]

    def _find_random_action(self, state):
        """
        Return a random action (here state) given that the agent is in state
        :param state:
        :return:
        """
        return random.choice(self.q_table[state])[0]

    def _save(self):
        """
        Save agent
        :return:
        """
        pickle.dump(self.q_table, open(self._q_table_path, "wb"))
        pickle.dump(self.replay_memory, open(self._mem_replay_path, "wb"))
        pickle.dump(self.best_accuracy, open(self._best_accuracy_path, "wb"))

    def q_learning(self, args):
        """
        Q Learning algorithm
        :return:
        """
        episode = self.episode
        while episode < self.M:
            episode += 1
            cprint("Episode {}".format(episode), color="green")
            found_new_architecture = False
            while not found_new_architecture:
                new_network_architecture = self._sample_new_network(self.epsilon)
                if not any([v[0] == new_network_architecture for v in self.replay_memory]):
                    found_new_architecture = True
            args.experiment_name = uuid.uuid4()
            args.layers = new_network_architecture
            # Train the model
            model = Model(args)
            try:
                model.build()
                accuracy = model.train(self)
            except Exception:
                import sys
                cprint("Exception raised:{}".format(sys.exc_info()[0]), color="red")
                accuracy = 0
            
            del model
            time.sleep(10)
            self._reduce_epsilon(episode)
            self.replay_memory.append((new_network_architecture, accuracy))
            if len(self.replay_memory) >= self.K:
                cprint("[*] Update Q values with replay", color="yellow", end="\n")
                indexes = sorted(random.sample(range(len(self.replay_memory)), self.K), reverse=True)
                for index in range(len(indexes)):
                    architecture, accuracy = self.replay_memory[index]
                    print(architecture, accuracy)
                    self.update_q_values(architecture, accuracy)
                cprint("Updated Q values!", color="green")

            # Save frequently
            self._save()
            

    def to_save(self, new_accuracy, experiment_name):
        cprint("[*] New model finished with accuracy {}".format(new_accuracy), color="green")

        # Not enough example, save everything
        if len(self.best_accuracy) < self.nb_to_keeps:
            cprint("\tlogs  and model saved", color="green")
            self.best_accuracy[experiment_name] = new_accuracy
            print("Len:{}, min:{}".format(len(self.best_accuracy), min(self.best_accuracy.values())))
            return True
        # Remove worst experience
        elif min(self.best_accuracy.values()) < new_accuracy:
            cprint("\tremove old bitch {} because got better is{}".format(
                min(self.best_accuracy.values()), new_accuracy), color="green")
            experiment_to_remove = min(self.best_accuracy, key=self.best_accuracy.get)
            del self.best_accuracy[experiment_to_remove]
            shutil.rmtree("logs/{}".format(experiment_to_remove))
            self.best_accuracy[experiment_name] = new_accuracy
            return True
        else:
            print("Len:{}, min:{}".format(len(self.best_accuracy), min(self.best_accuracy.values())))
            cprint("Not saving the model", color="red")
            shutil.rmtree("logs/{}".format(experiment_name))
            return False

    def update_q_values(self, arch, accuracy):
        def fi(state, new_state):
            return self.q_table[state].index([v for v in self.q_table[state] if v[0] == new_state][0])
        
        self.q_table[arch[-2]][fi(arch[-2], arch[-1])][1] = (1 - self.alpha) * self.q_table[arch[-2]][
            fi(arch[-2], arch[-1])][1] + self.alpha * accuracy
        
        for index in range(len(arch) - 3, -1, -1):
            self.q_table[arch[index]][fi(arch[index], arch[index + 1])][1] = (1 - self.alpha) \
                                                                             * self.q_table[arch[index]][
                                                                                 fi(arch[index], arch[index + 1])][1] \
                                                                             + self.alpha * \
                                                                               max([p[1] for p in
                                                                                    self.q_table[arch[index + 1]]])

    def _reduce_epsilon(self, episod):
        if episod > 1550:
            self.epsilon = 0.1
        elif episod > 1400:
            self.epsilon = 0.2
        elif episod > 1250:
            self.epsilon = 0.3
        elif episod > 1100:
            self.epsilon = 0.4
        elif episod > 950:
            self.epsilon = 0.5
        elif episod > 800:
            self.epsilon = 0.6
        elif episod > 700:
            self.epsilon = 0.7
        elif episod > 600:
            self.epsilon = 0.8
        elif episod > 500:
            self.epsilon = 0.9
        else:
            self.epsilon = 1


