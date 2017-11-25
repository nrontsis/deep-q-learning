# -*- coding: utf-8 -*-
import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import time
import argparse

EPISODES = 1000


class DQNAgent:
    def __init__(self, state_size, action_size, gamma=0.95, epsilon=1.0,
                 epsilon_min=0.01, epsilon_decay=0.995, learning_rate=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = gamma    # discount rate
        self.epsilon = epsilon  # exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


def main(save='evals/tmp.txt', **args):
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size, **args)
    # agent.load("./save/cartpole-dqn.h5")
    done = False
    batch_size = 32

    scores = deque(maxlen=100)
    for e in range(EPISODES):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        for i in range(500):
            # env.render()
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            reward = reward if not done else -10
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                # print("episode: {}/{}, score: {}, e: {:.2}"
                #       .format(e, EPISODES, i, agent.epsilon))
                break
        scores.append(i)
        mean_score = np.mean(scores)

        if len(agent.memory) > batch_size:
            agent.replay(batch_size)

        if mean_score >= 195 and e >= 100:
            print('Ran {} episodes. Solved after {} trials ✔'.format(e, e - 100))
            return e - 100

        if e % 100 == 0:
            print('[Episode {}] - Running average of survival is {} ticks.'.format(e, mean_score))

    print('Did not solve after {} episodes 😞'.format(e))
    return e


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gamma', type=float, help='Discount factor', default=0.95)
    parser.add_argument('-e', '--epsilon', type=float, help='Exploration rate',
                        default=1)
    parser.add_argument('-e_m', '--epsilon_min', type=float, help='Minimum exploration rate',
                        default=0.01)
    parser.add_argument('-e_d', '--epsilon_decay', type=float, help='Exploration rate decay',
                        default=0.995)
    parser.add_argument('-lr', '--learning_rate', type=float,
                        help='Learning rate',
                        default=0.001)
    parser.add_argument('-s', '--save', default='evals/tmp.txt')

    args = parser.parse_args()

    start = time.time()
    e = []
    for i in range(10):
        e.append(main(**vars(args)))
    end = time.time()
    print('Done in:', (end - start) / 60, 'mins.')
    final_score = np.mean(np.asarray(e))  # Final score (lower is better)
    print('Final score:', final_score)

    np.savetxt(args.save, np.array([final_score]))
