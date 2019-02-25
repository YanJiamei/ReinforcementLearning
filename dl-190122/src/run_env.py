# import dis_learning
# from RL_brain import DoubleDQN
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from Learner import Learner
from tool_func import load_datasets
import pandas as pd

if __name__ == "__main__":
    
    # import os
    # cuda = raw_input("CUDA DEVICES: ")
    # os.environ["CUDA_VISIBLE_DEVICES"] = cuda
    file_name = './logs/records-e-0.txt'
    # acc = 0.9888
    # with open(file_name , 'a') as file:
    #     file.write(str(acc))
    log_dir = './logs/gmm/'
    data_dir_1 = './datas/f-1.csv'
    # data_dir_2 = '../datas/e-1.csv'
    # data_dir_3 = '../datas/e-2.csv'
    # data_dir_4 = '../datas/e-3.csv'
    # data_all ='../datas/e-all.csv'
    data = [load_datasets(data_dir_1, tst_frac=0.2), load_datasets(data_dir_1, tst_frac=0.2)]
    # print(data[1])



    env = Learner(
        log_dir=log_dir,
        i_data=data[0]['train'],
        j_data=data[0]['train'],
        tst_data = data[0]['tst'],
        learning_steps = 500,
        learning_steps_max = 50000,
        hidden_layer_size=10,
        feature_size=3,
    )
    env.reset()
    for i in range(200):
        env.step(0)
        if i % 10 == 0:
            loss, acc = env.test(500)
            with open(file_name , 'a') as file:
                file.write('\nstep = ' + str(i*500))
                file.write('\nacc = ')
                file.write(str(acc))
                file.write(', loss = ')
                file.write(str(loss))

    env.destroy()
    