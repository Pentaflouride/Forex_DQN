import numpy as np
import tensorflow as tf
from collections import deque
import random

class ForexAgent:
    """
    A Deep Q-Network (DQN) agent for Forex trading.

    This agent uses a neural network to approximate the Q-function and make
    trading decisions based on the current market state.

    Attributes:
        state_size (int): The size of the input state vector.
        action_size (int): The number of possible actions.
        memory (deque): A replay buffer to store experiences.
        gamma (float): The discount factor for future rewards.
        epsilon (float): The exploration rate.
        epsilon_min (float): The minimum exploration rate.
        epsilon_decay (float): The decay rate for epsilon.
        learning_rate (float): The learning rate for the neural network.
        model (tf.keras.Model): The Q-network model.
        target_model (tf.keras.Model): The target Q-network model.
    """

    def __init__(self, state_size, action_size):
        """
        Initialize the Forex DQN agent.

        Args:
            state_size (int): The size of the input state vector.
            action_size (int): The number of possible actions.
        """
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0   # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _build_model(self):
        """
        Build a neural network model for Q-function approximation.

        Returns:
            tf.keras.Model: A compiled Keras model.
        """
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(24, input_dim=self.state_size, activation='relu'),
            tf.keras.layers.Dense(24, activation='relu'),
            tf.keras.layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate))
        return model

    def update_target_model(self):
        """
        Update the target model with weights from the main model.
        """
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        """
        Store an experience in the replay buffer.

        Args:
            state (numpy.array): The current state.
            action (int): The action taken.
            reward (float): The reward received.
            next_state (numpy.array): The resulting state.
            done (bool): Whether the episode has ended.
        """
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, has_position):
        state = state.astype('float32')
        if np.random.rand() <= self.epsilon:
            if has_position:
                # If the agent has bought, it can sell (2) or hold (0)
                return random.choice([0, 2])
            else:
                # If the agent has not bought, it can buy (1) or wait (3)
                return random.choice([1, 3])
        act_values = self.model.predict(state, verbose=0)
        if has_position:
            # Mask invalid actions when in position
            act_values[0][1] = -np.inf  # Cannot buy again
            act_values[0][3] = -np.inf  # Cannot wait when in position
        else:
            # Mask invalid actions when not in position
            act_values[0][2] = -np.inf  # Cannot sell without position
            act_values[0][0] = -np.inf  # Cannot hold without position
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        """
        Train the Q-network with experiences sampled from the replay buffer.

        Args:
            batch_size (int): The number of experiences to sample from the buffer.
        """
        minibatch = np.random.choice(len(self.memory), batch_size)
        for i in minibatch:
            state, action, reward, next_state, done = self.memory[i]
            state = np.array(state, dtype=np.float32)  # Ensure the correct data type
            next_state = np.array(next_state, dtype=np.float32)  # Ensure the correct data type
            target = self.model.predict(state, verbose = 0)
            if done:
                target[0][action] = reward
            else:
                t = self.target_model.predict(next_state, verbose = 0)[0]
                target[0][action] = reward + self.gamma * np.amax(t)
            self.model.fit(state, target, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        """
        Load the model weights from a file.

        Args:
            name (str): The name of the file to load the weights from.
        """
        self.model.load_weights(name)

    def save(self, name):
        """
        Save the model weights to a file.

        Args:
            name (str): The name of the file to save the weights to.
        """
        self.model.save_weights(name)
