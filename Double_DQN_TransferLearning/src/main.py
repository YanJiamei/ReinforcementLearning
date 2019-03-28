# import dis_learning
# from RL_brain import DoubleDQN
from PDDQN import *
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from learner_entropy_reward import *
from tool_func import *
import pandas as pd
import random as rnd
MEMORY_SIZE = 1000
ACTION_SPACE = 6




EPISODE_NUM = 20 # episodes that game reruns
BEGIN_LEARNING_STEP = 10 # after some begin steps, RL begin to learn
def train_DDQN(RL, env, file_name, dataset, trans_penalty=1.5, action_base=1.0, alpha=100):
    step = 0
    for episode in range(EPISODE_NUM):
        # trans_penalty = rnd.sample([0.8,1.0,1.2,1.4,1.6,1.8,2.0,2.2],1)[0]
        trans_penalty = rnd.sample([0.5,1.0,1.5,2.0],1)[0]
        observation = env.reset(trans_penalty=trans_penalty) #
        ac_num = 0
        action_count = []
        while True:
            action, random = RL.choose_action(observation)
            print('\naction:',action)
            if action>0:
                action_count.append([step,action,random])
                ac_num += 1
            # env take action and get next observation and reward    
            observation_, reward, done = env.step(action = action, trans_penalty= trans_penalty, action_base = action_base, alpha=alpha)
            # reward = reward*10
            print('reward: ',reward)
            # RL store (observation, action, reward, observation_) to retrain
            RL.store_transition(observation, action, reward, observation_)

            if (step > BEGIN_LEARNING_STEP):
                for i in range(20):
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
def tst_DDQN(RL, env, file_name, dataset, trans_penalty=0.8, action_base=2.0,alpha=100):
        observation = env.reset(trans_penalty=trans_penalty) #
        ac_num = 0
        action_count = []
        step=0
        BEGIN_LEARNING_STEP = 10
        while True:
            action, random = RL.choose_action(observation)
            print('\naction:',action)
            if action>0:
                #try unweighted
                # action=1
                action_count.append([action,random])
                ac_num += 1
            # env take action and get next observation and reward
            
            observation_, reward, done = env.step(action = action, trans_penalty= trans_penalty, action_base = action_base,alpha=alpha)
            print('reward: ',reward)
            # reward = reward*10

            RL.store_transition(observation, action, reward, observation_)
            if (step > BEGIN_LEARNING_STEP):
                for i in range(10):
	                RL.learn()
            loss, acc = env.test(dataset)
            localloss, localacc = env.test(env.local_data)
            print('observation: ', observation_)
            # ready for next step
            observation = observation_
            with open(file_name , 'a') as file:
                file.write('\nstep = ' + str(env.learn_step_counter) + '\n')
                file.write('\ntotal acc = ')
                file.write(str(acc))
                file.write('\ntotal loss = ')
                file.write(str(loss))
                file.write('\nlocal acc = ')
                file.write(str(localacc))
                file.write('\nlocal loss = ')
                file.write(str(localloss))
                file.write('\nobservation = ')
                file.writelines(str(observation))
            # break while loop when end of this episode
            if done:
                # print(loss_list)
                print('============================\n')
                print('\naction num: ', ac_num)
                print('\naction count: ', action_count)
                with open(file_name , 'a') as file:
                    file.write('\naction = ' + str(ac_num) + '\n')
                    file.writelines(str(action_count))
                    file.write('\n=============================================\n==========================================\n')
                break
            step+=1
def raw_tst(RL, env, file_name, dataset, action=0):
    env.reset()
    while True:
        observation_, reward, done = env.step(action = action, trans_penalty=0.1)
        loss, acc = env.test(dataset)
        localloss, localacc = env.test(env.local_data)
        with open(file_name , 'a') as file:
            file.write('\nstep = ' + str(env.learn_step_counter) + '\n')
            file.write('\ntotal acc = ')
            file.write(str(acc))
            file.write('\ntotal loss = ')
            file.write(str(loss))
            file.write('\nlocal acc = ')
            file.write(str(localacc))
            file.write('\nlocal loss = ')
            file.write(str(localloss))
        # break while loop when end of this episode
        if done:
            with open(file_name , 'a') as file:
                file.write('\n=============================================\n==========================================\n')
            break

if __name__ == "__main__":
    
    # import os
    # cuda = raw_input("CUDA DEVICES: ")
    # os.environ["CUDA_VISIBLE_DEVICES"] = cuda


    # log_dir = './ckpt'

    data = [load_datasets('./datas/50div10gmm10-1.csv', tst_frac=0), load_datasets('./datas/50div10gmm10-2.csv', tst_frac=0),\
                load_datasets('./datas/50div10gmm10-all.csv', tst_frac=0.8) ]
    datatrain = load_localdata('./datas/50div10gmm10-1.csv')
    with tf.variable_scope('DQN_with_prioritized_replay'):
        RL_prio = DQNPrioritizedReplay(
            n_actions=5, #trans weights
            n_features=4,
            learning_rate=0.001, #RL learning rate
            reward_decay=0.98, #reward decay --the infuluence of reward
            e_greedy=0.90,#max e-greedy
            e_greedy_increment=0.001, 
            replace_target_iter=80,
            memory_size=400,
            batch_size=20,

            output_graph=False,
            prioritized=False,
            dueling=False,
            sess=None, #for reload
    )
    model_file = tf.train.latest_checkpoint('./ckpt/')
    tf.train.Saver().restore(RL_prio.sess, model_file)
    env = Learner_entro_reward(
        # log_dir=log_dir,
        i_data=datatrain['idata'],
        j_data=datatrain['jdata'],
        # i_data=data[0]['train'],
        # j_data=data[1]['train'],
        learning_steps = 200,
        learning_steps_max = 20000,
        feature_size = 50,
        # batch_size=20,
        # buffer_size=20
    )
    file_name = './records/h5-reward2-pen0.5-2.0-lr0.001-mem400.txt'
    with open(file_name , 'a') as file:
        file.write('\n\n===NEW DATA LOADING====\n\n')
        file.write('\n\ni_data: '+str(0))
        file.write('\n\nj_data: '+str(0))
    # RL_prio.plot_cost()
    # saver = tf.train.Saver()
    # model_file = tf.train.latest_checkpoint('./ckpt/')
    # saver.restore(RL_prio.sess, model_file)

    train_DDQN(RL = RL_prio, env = env, file_name = file_name, \
                dataset = datatrain['alldata'],\
                # dataset = data[2]['train'],
                trans_penalty=1.0, action_base=1.0,
                alpha=500)
    with open(file_name,'a') as file:
        file.write('\ncost\n')
        file.writelines(str(RL_prio.cost_his))    
        file.write('\n=============================================\n==========================================\n')
    
    RL_prio.save_model('./ckpt/h5')

    # env.i_data = data[0]['train']
    # env.j_data = data[1]['train']
    # file_name = './records/tst/f7-retrain-5w-tst-pen1.5.txt'
    # tst_DDQN(RL = RL_prio, env = env, file_name = file_name, dataset = data[2]['tst'],\
    #         trans_penalty=1.5, action_base=1.0)
    
    RL_prio.plot_cost()


    # with open(file_name,'a') as file:
    #     file.write('\n===================\n==========================\n\ndqn tst\n\n====================\n======================\n')
    # env_tst = Learner_entro_reward(
    #     log_dir=log_dir,
    #     i_data=data[1]['train'],
    #     j_data=data[0]['train'],
    #     learning_steps = 500,
    #     learning_steps_max = 50000,
    #     feature_size = 50,
    # )
    # tst_DDQN(RL = RL_prio, env = env_tst, file_name = file_name, dataset = data[2]['train'],\
    #             trans_penalty=1.0, action_base=1.0)



    # with open(file_name,'a') as file:
    #     file.write('\n===================\n==========================\n\nraw tst: action=0\n\n====================\n======================\n')
    # env_tst = Learner_entro_reward(
    #     log_dir=log_dir,
    #     i_data=data[1]['train'],
    #     j_data=data[0]['train'],
    #     learning_steps = 500,
    #     learning_steps_max = 50000,
    #     feature_size = 50,
    # )
    # raw_tst(RL = RL_prio, env = env_tst, file_name = file_name, dataset = data[2]['train'], action=0)



    # with open(file_name,'a') as file:
    #     file.write('\n===================\n==========================\n\nraw tst: action=1\n\n====================\n======================\n')
    # env_tst = Learner_entro_reward(
    #     log_dir=log_dir,
    #     i_data=data[1]['train'],
    #     j_data=data[0]['train'],
    #     learning_steps = 500,
    #     learning_steps_max = 50000,
    #     feature_size = 50,
    # )
    # raw_tst(RL = RL_prio, env = env_tst, file_name = file_name, dataset = data[2]['train'], action=1)

    # RL_prio.plot_cost()
    # env.destroy()
    
