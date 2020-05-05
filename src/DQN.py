import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, layers
from collections import deque
import random
import gym
import src.utils as utils


# from src.agent import Agent

class QNet(Model):
    def __init__(self, name, n_observations, hidden_dims, n_actions):
        """
            Creates a densely connected multi-layer neural network.
            Args:
                n_observations: Size of observations
                n_actions: Size of the actions
                hidden_dims: A list of dimensions for hidden layers
        """

        super(QNet, self).__init__(name=name)
        self.input_layer = layers.InputLayer(input_shape=(n_observations,))
        self.hidden = []
        for dim in hidden_dims:
            self.hidden.append(layers.Dense(dim, activation='relu', kernel_initializer='RandomNormal'))
        self.logits = layers.Dense(n_actions, kernel_initializer='RandomNormal')

    @tf.function
    def call(self, inputs):
        hidden = self.input_layer(inputs)
        for fc in self.hidden:
            hidden = fc(hidden)
        return self.logits(hidden)


class DQN:
    def __init__(self, name, n_observations, n_actions, hidden_dims, n_batch=32,
                 min_experiences=50, max_experiences=1000,
                 gamma=0.99, learning_rate=.0015, epsilon=.9, min_epsilon=.01, epsilon_decay=.995):
        """
          Deep Q-Network Agent.

          Args:
            name: The name of the agent
            n_observations: Size of observations
            n_actions: Size of the actions
            hidden_dims: A list of dimensions for hidden layers
            n_batch: Batch size
            min_experiences: The Step to begin learning
            max_experiences: The maximum size of the replay buffer
            gamma: Discount rate
            learning_rate: Learning rate
            epsilon: Exploration rate
            epsilon_decay: Epsilon decay rate
            min_epsilon: Minimum value of epsilon
        """

        self.name = name
        self.n_observations = n_observations
        self.n_actions = n_actions
        self.n_batch = n_batch
        self.min_experiences = min_experiences
        self.gamma = gamma
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)
        self.model = QNet("QNet", self.n_observations, hidden_dims, self.n_actions)
        self.replay_buffer = deque(maxlen=max_experiences)
        self.min_epsilon = min_epsilon
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.frames = []

    def predict(self, inputs):
        return self.model(np.atleast_2d(inputs.astype('float32')))

    def act(self, observation, epsilon=0):
        if epsilon > 0 and np.random.random() < epsilon:
            return np.random.choice(self.n_actions)
        return np.argmax(self.predict(observation)[0])

    def remember(self, observation, action, reward, next_observation, done):
        self.replay_buffer.append((observation, action, reward, next_observation, done))

    def memory_check(self):
        return len(self.replay_buffer) >= self.min_experiences

    # @tf.function
    def replay_and_train(self, target_net):
        minibatch = random.sample(self.replay_buffer, self.n_batch)
        observations, actions, rewards, next_observations, dones = list(zip(*minibatch))
        observations = np.asarray(observations)
        actions = np.asarray(actions)
        rewards = np.asarray(rewards)
        next_observations = np.asarray(next_observations)
        dones = np.asarray(dones)

        next_values = tf.math.reduce_max(target_net.predict(next_observations), axis=1)
        target_values = tf.where(dones,
                                 tf.convert_to_tensor(rewards, dtype=tf.float32),
                                 tf.convert_to_tensor(rewards + self.gamma * next_values))

        with tf.GradientTape() as tape:
            predicted_values = tf.math.reduce_sum(
                self.predict(observations) * tf.one_hot(actions, self.n_actions), axis=1)
            loss = tf.math.reduce_sum(tf.square(target_values - predicted_values))
        variables = self.model.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))

        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay

        return loss

    def copy(self, agent):
        self.model.set_weights(agent.model.get_weights())

    def train(self, env, target, copy_steps=100, cum_steps=0):
        rewards = 0
        done = False
        observations = env.reset()
        while not done:
            action = self.act(observations, self.epsilon)
            prev_observations = observations
            observations, reward, done, _ = env.step(action)
            rewards += reward

            self.remember(prev_observations, action, reward, observations, done)
            if self.memory_check():
                self.replay_and_train(target)
            cum_steps += 1

            if cum_steps % copy_steps == 0:
                target.copy(self)

        env.reset()

        return cum_steps, rewards

    def display(self, env):
        self.frames = []
        observation = env.reset()
        done = False

        while not done:
            env.render()
            action = self.act(observation, 0)
            observation, reward, done, _ = env.step(action)

        utils.save_gif(self.frames, 'assets/gif/{name}.gif'.format(name=self.name))
        utils.display_gif('assets/gif/{name}.gif'.format(name=self.name))
        self.frames = []
