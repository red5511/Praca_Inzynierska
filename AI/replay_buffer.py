import numpy as np

class ReplayBuffer():
    def __init__(self, size, BUFFER_BATCH, resume_training, SHAPE=8, N=0, ITERS=0, FRAMES_PER_STATE=4):
        if resume_training:
            if N == 0 or ITERS == 0:
                print("N albo ITERS nie moze byc 0")
                raise ValueError('......')  

            self.actions = np.load('saved_buffer/actions{}.npz'.format(N))['arr_0']
            self.rewards = np.load('saved_buffer/rewards{}.npz'.format(N))['arr_0']
            self.terminal = np.load('saved_buffer/terminal{}.npz'.format(N))['arr_0']
            self.states = np.load('saved_buffer/states{}.npz'.format(N))['arr_0']
            self.new_states = np.load('saved_buffer/new_states{}.npz'.format(N))['arr_0']

            if size < ITERS:
                self.ready_to_train = True
                self.full_buffer = True
            else:
                self.ready_to_train = False
                self.full_buffer = False

            print("FULL_BUFFER = ", self.full_buffer)

            self.counter = int(ITERS) % size
            self.size = size
            self.BUFFER_BATCH = BUFFER_BATCH

        else:
            self.actions = np.zeros(size, dtype=np.int8)#8-bit signed integer (-128 to 127).
            self.rewards = np.zeros(size, dtype=np.int8)
            self.terminal = np.zeros(size, dtype=np.bool)
            self.states = np.zeros((size, SHAPE), dtype=np.int8)
            self.new_states = np.zeros((size, SHAPE), dtype=np.int8)
            self.ready_to_train = False
            self.full_buffer = False
            self.counter = 0
            self.BUFFER_BATCH = BUFFER_BATCH
            self.size = size

    def append(self, state, new_state, reward, action, terminal):
        self.states[self.counter] = state
        self.new_states[self.counter] = new_state
        self.rewards[self.counter] = reward
        self.actions[self.counter] = action
        self.terminal[self.counter] = terminal

        if self.counter + 1 == self.size:
            self.full_buffer = True


        self.counter = (self.counter + 1) % self.size

        if not self.ready_to_train:
            if self.counter >= self.BUFFER_BATCH:
                self.ready_to_train = True

        

    def random_samples(self, batch_size):
        if self.full_buffer:
            batch = np.random.choice(self.size, batch_size)
        elif self.ready_to_train:
            batch = np.random.choice(self.counter, batch_size)


        states = self.states[batch]
        new_state = self.new_states[batch]
        actions = self.actions[batch]
        rewards = self.rewards[batch]
        terminal = self.terminal[batch]

        return [states, new_state, actions, rewards, terminal]
