from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
import numpy as np

def bulid_model(SHAPE, lr, env):
    model = models.Sequential()

    model.add(layers.Dense(64, activation="relu", input_shape=(SHAPE,)))
    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dense(env.action_n, activation='linear'))
    model.compile(loss='mse', optimizer=Adam(learning_rate=lr))

    return model

class Agent():
    def __init__(self, env, buffer, lr, gamma, resume_training, N=0, SHAPE=8):
        if resume_training:
            if N == 0:
                print('N niemoze byc 0')
                raise ValueError('......')


            self.train_model = models.load_model('saved_models/model{}.h5'.format(N))
            self.target_model = models.load_model('saved_models/model{}.h5'.format(N))

        else:
            self.train_model = bulid_model(SHAPE, lr, env)
            self.target_model = bulid_model(SHAPE, lr, env)

            self.target_model.set_weights(self.train_model.get_weights())
        
        self.replay_buffer = buffer
        self.env = env
        self.gamma = gamma
        self.epsilon_min = 0.01
        self.ACTIONS = ["LEFT", "RIGHT", "SHOOT", "NONE"]


    def make_move(self, state, eps):
        eps_legit = max(eps, self.epsilon_min)
        if np.random.rand(1) < eps_legit:
            action = self.env.sample()
            return self.ACTIONS.index(action)
        else:
            state = state.reshape((1, 6))
            action = self.train_model.predict(state)[0]
            return np.argmax(action)

    def learn(self, BUFFER_BATCH):
        if self.replay_buffer.ready_to_train:

            states, new_states, action, reward, done =  self.replay_buffer.random_samples(BUFFER_BATCH)
            states = states
            new_states = new_states

            targets = self.train_model.predict(states)
            new_state_targets = self.target_model.predict(new_states)

            for i in range(len(states)):
                target = targets[i]
                if done[i]:
                    target[action[i]] = reward[i]
                else:
                    Q_future = max(new_state_targets[i])
                    target[action[i]] = reward[i] + Q_future * self.gamma


            self.train_model.fit(states, targets, epochs=1, verbose=0)