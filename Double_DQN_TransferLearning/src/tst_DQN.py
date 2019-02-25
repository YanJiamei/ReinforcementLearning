from PDDQN import *
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from learner_entropy_reward import *
from tool_func import load_datasets
import pandas as pd

def tst_DDQN(RL, env, file_name, dataset, trans_penalty=0.8, action_base=2.0):
        observation = env.reset() #
        ac_num = 0
        action_count = []
        while True:
            action, random = RL.choose_action(observation)
            print('\naction:',action)
            if action>0:
                action_count.append([action,random])
                ac_num += 1
            # env take action and get next observation and reward    
            observation_, reward, done = env.step(action = action, trans_penalty= trans_penalty, action_base = action_base)
            print('reward: ',reward)
            reward = reward*10
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
    
    with tf.variable_scope('DQN_with_prioritized_replay'):
        RL_prio = DQNPrioritizedReplay(
            n_actions=6, #trans weights
            n_features=6,
            learning_rate=0.01, #RL learning rate
            reward_decay=0.99, #reward decay --the infuluence of reward
            e_greedy=1,#max e-greedy
            e_greedy_increment=None, 
            replace_target_iter=20,
            memory_size=800,
            batch_size=20,

            output_graph=False,
            prioritized=False,
            dueling=False,
            sess=None, #for reload
    )
    # saver = tf.train.import_meta_graph(./ckpt)
    model_file = tf.train.latest_checkpoint('./ckpt/')
    tf.train.Saver().restore(RL_prio.sess, model_file)
    data = [load_datasets('./datas/50div10gmm10-1.csv', tst_frac=0), load_datasets('./datas/50div10gmm10-2.csv', tst_frac=0),\
            load_datasets('./datas/50div10gmm10-all.csv', tst_frac=0.8) ]
    env_tst = Learner_entro_reward(
            # log_dir=log_dir,
            i_data=data[1]['train'],
            j_data=data[0]['train'],
            learning_steps = 500,
            learning_steps_max = 50000,
            feature_size = 50,
        )

    file_name = './records/tst/13-tst-decay1.0.txt'
    tst_DDQN(RL = RL_prio, env = env_tst, file_name = file_name, dataset = data[2]['train'],\
            trans_penalty=1.8, action_base=1.0)
