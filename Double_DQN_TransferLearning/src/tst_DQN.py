from PDDQN import *
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from learner_entropy_reward import *
from tool_func import load_datasets
import pandas as pd

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
            # if (step > BEGIN_LEARNING_STEP):
            #     for i in range(10):
	        #         RL.learn()
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

# saver = tf.train.Saver()
# model_file = tf.train.latest_checkpoint('./logs/')
# saver.restore(RL_prio.sess, model_file)
# RL_prio = DQNPrioritizedReplay(
#     n_actions=6, #trans weights
#     n_features=6,
#     learning_rate=0.01, #RL learning rate
#     reward_decay=0.99, #reward decay --the infuluence of reward
#     e_greedy=0.95,#max e-greedy
#     e_greedy_increment=0.005, 
#     replace_target_iter=20,
#     memory_size=800,
#     batch_size=20,

#     output_graph=False,
#     prioritized=False,
#     dueling=False,
#     sess=None, #for reload
# )
# # model_file = './ckpt/checkpoint'
# model_file = tf.train.latest_checkpoint('./ckpt/')
# tf.train.Saver().restore(RL_prio.sess, model_file)
# env_tst = Learner_entro_reward(
#         # log_dir=log_dir,
#         i_data=data[1]['train'],
#         j_data=data[0]['train'],
#         learning_steps = 500,
#         learning_steps_max = 50000,
#         feature_size = 50,
#     )
# file_name = './records/tst/1.txt'
# tst_DDQN(RL = RL_prio, env = env_tst, file_name = file_name, dataset = data[2]['train'],\
#             trans_penalty=1.0, action_base=1.0)

if __name__ == "__main__":
    
    data = [load_datasets('./datas/50div10gmm10-1.csv', tst_frac=0), load_datasets('./datas/50div10gmm10-2.csv', tst_frac=0),\
                load_datasets('./datas/50div10gmm10-all.csv', tst_frac=0.8) ]

    with tf.variable_scope('DQN_with_prioritized_replay'):
        RL_prio = DQNPrioritizedReplay(
            n_actions=5, #trans weights
            n_features=4,
            # learning_rate=0.01, #RL learning rate
            learning_rate=0.005, #retrain learning rate
            reward_decay=0.98, #reward decay --the infuluence of reward
            e_greedy=0.95,#max e-greedy
            e_greedy_increment=None, 
            replace_target_iter=20,
            memory_size=100,
            batch_size=10,

            output_graph=False,
            prioritized=True,
            dueling=False,
            sess=None, #for reload
    )
    env = Learner_entro_reward(
        # log_dir=log_dir,
        i_data=data[0]['train'],
        j_data=data[1]['train'],
        learning_steps = 200,
        learning_steps_max = 50000,
        feature_size = 50,
    )
    model_file = tf.train.latest_checkpoint('./ckpt/')
    tf.train.Saver().restore(RL_prio.sess, model_file)

    file_name = './records/tst/g3-nonretrain-tst-pen1.6.txt'
    tst_DDQN(RL = RL_prio, env = env, file_name = file_name, dataset = data[2]['tst'],\
            trans_penalty=1.6, action_base=1.0)

    # RL_prio.plot_cost()