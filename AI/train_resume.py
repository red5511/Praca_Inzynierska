import sys
sys.path.append('../')
import numpy as np
import random
from air_hockey_env import Air_hockey_env
from replay_buffer import ReplayBuffer
from agent import Agent
import json
from numpy import save
from numpy import load
from numpy import savez_compressed

ACTIONS = ["LEFT", "RIGHT", "SHOOT", "NONE"]



def play_episode(agent1, agent2, buffer1, buffer2, env, eps, gamma, BUFFER_BATCH, episode_number):
    state1, state2 = env.reset()
    done = False
    episode_reward1 = 0
    iter = 0
        
    while not done:

        action1 = agent1.make_move(state1, eps)
        action2 = agent2.make_move(state2, eps)

        new_state1, new_state2, reward1, reward2, done = env.step(ACTIONS[action1], ACTIONS[action2])
        episode_reward1 += reward1


        buffer1.append(state1, new_state1, reward1, action1, done)
        buffer2.append(state2, new_state2, reward2, action2, done)


        state1 = new_state1
        state2 = new_state2

        agent1.learn(BUFFER_BATCH) # uczenie sieci, przewidywania rewardow na podstawie stanu w ktorym sie znajdujemy
        agent2.learn(BUFFER_BATCH) 
        iter += 1
        if done:
            print("Iter =", iter)
            

    agent1.target_model.set_weights(agent1.train_model.get_weights())
    agent2.target_model.set_weights(agent2.train_model.get_weights())


    return episode_reward1, iter


X, Y = 60, 30


env = Air_hockey_env(X, Y)
EPISODES_NUMBER = 12_500 + 1
BUFFER_SIZE = 200_000 
BUFFER_BATCH = 32
gamma = 0.99
learingRate = 0.001
eps_min = 0.01
eps_decay = 0.013
RESUME_TRAINING = 7450
SHAPE = 6# p1.x, p1.y, b.x, b.y, b.dx, b.dy



print("Wczytuje arraye")
buf_rewards = np.load('plot_data/rewardss{}.npz'.format(RESUME_TRAINING))
rewards = buf_rewards['arr_0']

buf_iters = np.load('plot_data/iters{}.npz'.format(RESUME_TRAINING))
iters = buf_iters['arr_0']
print('Iters sum', np.sum(iters), "1_" + str(RESUME_TRAINING))

if iters.shape[0] < EPISODES_NUMBER:
    iters = np.pad(iters, (0, EPISODES_NUMBER - iters.shape[0]), mode='constant', constant_values=0)
    rewards = np.pad(rewards, (0, EPISODES_NUMBER - rewards.shape[0]), mode='constant', constant_values=0)


buffer1 = ReplayBuffer(BUFFER_SIZE, BUFFER_BATCH, True, SHAPE, "1_" + str(RESUME_TRAINING), np.sum(iters))
buffer2 = ReplayBuffer(BUFFER_SIZE, BUFFER_BATCH, True, SHAPE, "2_" + str(RESUME_TRAINING), np.sum(iters))

agent1 = Agent(env, buffer1, learingRate, gamma, True, "1_" + str(RESUME_TRAINING))
agent2 = Agent(env, buffer2, learingRate, gamma, True, "2_" + str(RESUME_TRAINING))
print("Arraye wczytane")


eps = 0.1
for i in range(RESUME_TRAINING + 1, EPISODES_NUMBER):
    # eps = 1.0/(0.1*n+1)

    if i % 50 == 0 and i > 0:
        render1, render2 = env.render()
        with open('renders/player_{}.json'.format(i), 'w') as f:
            json.dump(render1, f)
        with open('renders/player2_{}.json'.format(i), 'w') as f:
            json.dump(render2, f)

    print('Episode', i, 'Epsilon', max(eps, eps_min))
    
    rewards[i], iters[i] = play_episode(agent1, agent2, buffer1, buffer2, env, eps, gamma, BUFFER_BATCH, i)

    eps -= eps_decay

    if i % 50 == 0 and i > 0:
        print('Episode', i, 'Epsilon', eps, 'Reward', rewards[i], 'Iteracje', iters[i])
        print(rewards[i-10:i])
        print("Srednia nagrod dla ostatnich 10 ep:", rewards[i-10:i].mean())
        print("Srednia iteracji dla ostatnich 10 ep:", iters[i-10:i].mean())

    if i % 50 == 0 and i > 0:
        agent1.train_model.save('saved_models/model1_{}.h5'.format(i))
        agent2.train_model.save('saved_models/model2_{}.h5'.format(i))

        savez_compressed('saved_buffer/states1_{}.npz'.format(i), buffer1.states)
        savez_compressed('saved_buffer/states2_{}.npz'.format(i), buffer2.states)

        savez_compressed('saved_buffer/new_states1_{}.npz'.format(i), buffer1.new_states)
        savez_compressed('saved_buffer/new_states2_{}.npz'.format(i), buffer2.new_states)

        savez_compressed('saved_buffer/rewards1_{}.npz'.format(i), buffer1.rewards)
        savez_compressed('saved_buffer/rewards2_{}.npz'.format(i), buffer2.rewards)

        savez_compressed('saved_buffer/actions1_{}.npz'.format(i), buffer1.actions)
        savez_compressed('saved_buffer/actions2_{}.npz'.format(i), buffer2.actions)

        savez_compressed('saved_buffer/terminal1_{}.npz'.format(i), buffer1.terminal)
        savez_compressed('saved_buffer/terminal2_{}.npz'.format(i), buffer2.terminal)

        savez_compressed('plot_data/rewardss{}.npz'.format(i), rewards)
        savez_compressed('plot_data/iters{}.npz'.format(i), iters)