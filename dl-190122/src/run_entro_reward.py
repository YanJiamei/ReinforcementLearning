# import dis_learning
# from RL_brain import DoubleDQN
from PDDQN import *
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from learner_entropy_reward import *
from tool_func import load_datasets
import pandas as pd
MEMORY_SIZE = 1000
ACTION_SPACE = 6




EPISODE_NUM = 60 # episodes that game reruns
BEGIN_LEARNING_STEP = 20 # after some begin steps, RL begin to learn
def train_DDQN(RL, env, file_name, dataset, trans_penalty=0.8, action_base=2.0):
    step = 0
    for episode in range(EPISODE_NUM):
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
            observation_, reward, done = env.step(action = action, trans_penalty= trans_penalty, action_base = action_base)
            print('reward: ',reward)
            reward = reward*10
            # RL store (observation, action, reward, observation_) to retrain
            RL.store_transition(observation, action, reward, observation_)

            if (step > BEGIN_LEARNING_STEP):
                for i in range(10):
	            RL.learn()
            if step % 10 == 0:
                loss, acc = env.test(dataset)
                print('observation: ', observation_)
            # ready for next step
            observation = observation_

            # break while loop when end of this episode
            if done:
                # print(loss_list)
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
                    file.write('\nobservation = ')
                    file.writelines(str(observation))

                break
            

            step += 1



if __name__ == "__main__":
    
    # import os
    # cuda = raw_input("CUDA DEVICES: ")
    # os.environ["CUDA_VISIBLE_DEVICES"] = cuda
    file_name = './logs/records/nonprio-rewarx10-penal0.8-ep60-ac5x2.0.txt'
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

    data = [load_datasets('./datas/50div10gmm10-1.csv', tst_frac=0), load_datasets('./datas/50div10gmm10-2.csv', tst_frac=0),\
                load_datasets('./datas/50div10gmm10-all.csv', tst_frac=0.8) ]
    # with tf.variable_scope('Double_DQN'):
    #     double_DQN = DoubleDQN(
    #             n_actions=ACTION_SPACE,
    #             n_features=5,
    #             learning_rate=0.01,
    #             reward_decay=0.95,
    #             e_greedy=0.95,
    #             replace_target_iter=10,
    #             memory_size=10000,
    #             batch_size=20,
    #             e_greedy_increment=0.01,
    #             output_graph=True,
    #             double_q=True,
    #             sess=None,
    #     )
    with tf.variable_scope('DQN_with_prioritized_replay'):
        RL_prio = DQNPrioritizedReplay(
            n_actions=5,
            n_features=6,
            learning_rate=0.01,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=20,
            memory_size=1000,
            batch_size=20,
            e_greedy_increment=0.005,
            output_graph=True,
            prioritized=False,
            dueling=False,
            sess=None,
    )
    env = Learner_entro_reward(
        log_dir=log_dir,
        i_data=data[1]['train'],
        j_data=data[0]['train'],
        learning_steps = 1000,
        learning_steps_max = 100000,
        feature_size = 50,
    )
    with open(file_name , 'a') as file:
        file.write('\n\n===NEW DATA LOADING====\n\n')
        file.write('\n\ni_data: '+str(1))
        file.write('\n\nj_data: '+str(0))

    train_DDQN(RL = RL_prio, env = env, file_name = file_name, dataset = data[2]['train'])
    with open(file_name,'a') as file:
        file.write('\ncost\n')
        file.writelines(str(RL_prio.cost_his))    
    
	# double_DQN.plot_cost()
        # j = np.random.randint(4)
        # env.i_data = data[j]['train']
        # env.reward_tst_data = data[j]['tst']

        # k = np.random.randint(4)
        # env.j_data = data[k]['train']
        # with open(file_name , 'a') as file:
        #     file.write('\n\n===NEW DATA LOADING====\n\n')
        #     file.write('\n\ni_data: '+str(j))
        #     file.write('\n\nj_data: '+str(k))
    
    # double_DQN.plot_cost()
    RL_prio.save_model('dddddddd')
    # env.destroy()
    
