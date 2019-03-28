from PDDQN import *
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from learner_fortst import Learner
from tool_func import load_datasets
import pandas as pd

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
env_1 = Learner(
    local_data=data[1]['train'],
    buffer_data=data[0]['train'],
    learning_steps=500,
    learning_steps_max=50000,
    feature_size=50,
    )

env_2 = Learner(
    # log_dir=log_dir,
    local_data=data[0]['train'],
    buffer_data=data[1]['train'],
    learning_steps=500,
    learning_steps_max=50000,
    feature_size=50,
    )

env_3 = Learner(
    # log_dir=log_dir,
    local_data=data[1]['train'],
    buffer_data=data[0]['train'],
    learning_steps=500,
    learning_steps_max=50000,
    feature_size=50,
    )
observation1 = env_1.reset()
observation2 = env_2.reset()
observation3 = env_3.reset()
for i in range(TRAINING_STEPS):
    
    action, _ = RL_prio.choose_action(observation1)
    env_1.reload_buffer(action,buff_data)
    observation1, reward, done = env_1.step(trans_penalty=1.5,action_base=1.0)