# import dis_learning
# from RL_brain import DoubleDQN
from PDDQN import *
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from Learner import Learner
from tool_func import load_datasets
import pandas as pd
MEMORY_SIZE = 10000
ACTION_SPACE = 5




EPISODE_NUM = 5 # episodes that game reruns
BEGIN_LEARNING_STEP = 20 # after some begin steps, RL begin to learn
def train_DDQN(RL, env, file_name, penalty, tst_data):
    step = 0
    for episode in range(EPISODE_NUM):
        env.test(tst_data)
        observation = env.reset() #
        ac_num = 0
        action_count = []
        while True:
            action, random = RL.choose_action(observation)
            print('\naction:',action)
            if action>0:
                action_count.append([step,action,random])
                ac_num += 1
            # env take action and get next observation and reward    
            observation_, reward, done= env.step(action = action, trans_penalty= penalty)
            print('reward: ',reward)
            
            # RL store (observation, action, reward, observation_) to retrain
            RL.store_transition(observation, action, reward, observation_)

            if (step > BEGIN_LEARNING_STEP):
                RL.learn()
            if step % 10 == 0:
                loss,acc = env.test(tst_data)
                print(observation_)
            # ready for next step
            observation = observation_

            # break while loop when end of this episode
            if done:
                print(loss_list)
                print('\n============================episode = ', episode)
                print('============================\n')
                print('\naction num: ', ac_num)
                print('\naction count: ', action_count)
                with open(file_name , 'a') as file:
                    file.write('\n==========================================================================================')
                    file.write('\nepisode = ' + str(episode))
                    file.write('\naction = ' + str(ac_num) + '\n')
                    file.writelines(str(action_count))
                    file.write('\ntotal acc = ')
                    file.write(str(acc))
                    file.write('\ntotal loss = ')
                    file.write(str(loss))

                break
            

            step += 1



if __name__ == "__main__":
    
    # import os
    # cuda = raw_input("CUDA DEVICES: ")
    # os.environ["CUDA_VISIBLE_DEVICES"] = cuda
    file_name = './logs/records-f.txt'
    # acc = 0.9888
    # with open(file_name , 'a') as file:
    #     file.write(str(acc))
    log_dir = './logs/gmm/'
    # data_dir_1 = './datas/10in15_50div50_100-0.csv'
    # data_dir_2 = './datas/10in15_50div50_100-1.csv'
    # data_dir_3 = './datas/10in15_50div50_100-2.csv'
    # data_dir_4 = './datas/10in15_50div50_100-3.csv'
    # data_all ='./datas/d-all.csv'
    # data = [load_datasets(data_dir_1, tst_frac=0.1), load_datasets(data_dir_2, tst_frac=0.1), load_datasets(data_dir_3, tst_frac=0.1), \
    #                         load_datasets(data_dir_4, tst_frac=0.1), load_datasets(data_all, tst_frac=0.1)]
    # print(data[1])

    data = [load_datasets('./datas/f-1.csv', tst_frac=0.1), load_datasets('./datas/f-2.csv', tst_frac=0.1)]

    with tf.variable_scope('DQN_with_prioritized_replay'):
        RL_prio = DQNPrioritizedReplay(
            n_actions=ACTION_SPACE,
            n_features=5,
            learning_rate=0.005,
            reward_decay=0.9,
            e_greedy=0.99,
            replace_target_iter=20,
            memory_size=10000,
            batch_size=20,
            e_greedy_increment=0.005,
            output_graph=False,
            prioritized=True,
            sess=None,
    )
    env = Learner(
        log_dir=log_dir,
        i_data=data[1]['train'],
        j_data=data[0]['train'],
        tst_data = data[1]['tst'],
        learning_steps = 500,
        learning_steps_max = 50000,   
    )
    with open(file_name , 'a') as file:
        file.write('\n\n===NEW DATA LOADING====\n\n')
        file.write('\n\ni_data: '+str(1))
        file.write('\n\nj_data: '+str(3))
    for i in range(5):
        train_DDQN(RL = RL_prio, env = env, file_name = file_name, penalty=0.02)
        # double_DQN.plot_cost()
        j = np.random.randint(4)
        env.i_data = data[j]['train']
        env.reward_tst_data = data[j]['tst']

        k = np.random.randint(4)
        env.j_data = data[k]['train']
        with open(file_name , 'a') as file:
            file.write('\n\n===NEW DATA LOADING====\n\n')
            file.write('\n\ni_data: '+str(j))
            file.write('\n\nj_data: '+str(k))
    
    double_DQN.plot_cost()
    double_DQN.save_model('episode-3')
    env.destroy()
    