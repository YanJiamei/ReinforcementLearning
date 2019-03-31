from LearnerEnv import *
from tool_func import load_weighted_datasets
from PDDQN import *
import tensorflow as tf
from MajorVote import *

def EnvStepOn(RL, Env, Env_i, Env_j, observation_i, observation_j, buff_i, penal_i, buff_j, penal_j):    
    print('\nobs.:')
    print('i:', observation_i)
    print('j:', observation_j)
    action_i, q_i_ , _i = RL.choose_action(observation_i)
    q_i = q_i_[0][action_i]
    print('action_1: ', action_i, '\tq_i: ', q_i, _i)
    action_j, q_j_ , _j = RL.choose_action(observation_j)
    q_j = q_j_[0][action_j]
    print('action_2: ', action_j, '\tq_j: ', q_j, _j)
    action_num = 0
    if(action_i>0 and action_j>0):
        action_num=1
        if(q_i>q_j):
            print('choosing iiiii...')
            Env.step1_action(action_i, buff_i, penal_i)
            reward = Env.step2_train_and_get_reward()
            print('reward:', reward)
            buff_i = Env_i.local_data.sample(n=BUFF_SIZE)
        else:
            print('choosing jjj...')
            Env.step1_action(action_j, buff_j, penal_j)
            reward = Env.step2_train_and_get_reward()
            print('reward:', reward,'\n')
            buff_j = Env_j.local_data.sample(n=BUFF_SIZE)
    elif(action_i!=0):
        action_num=1
        print('choosing iiiii...')
        Env.step1_action(action_i, buff_i, penal_i)
        reward = Env.step2_train_and_get_reward()
        print('reward:', reward)
        buff_i = Env_i.local_data.sample(n=BUFF_SIZE)
    elif(action_j!=0):
        action_num=1
        print('choosing jjj...')
        Env.step1_action(action_j, buff_j, penal_j)
        reward = Env.step2_train_and_get_reward()
        print('reward:', reward,'\n')
        buff_j = Env_j.local_data.sample(n=BUFF_SIZE)
    else:
        print('no trans...')
        Env.step1_action(0, buff_j, penal_j)
        reward = Env.step2_train_and_get_reward()
        print('reward:', reward,'\n')
    observation_i = Env.get_observation(buff_i, penal_i)
    observation_j = Env.get_observation(buff_j, penal_j)
    return observation_i, observation_j, buff_i, buff_j, action_num

if __name__ == "__main__":
    data_dir_1 = './datas/50div10gmm10/d1.csv'
    data_dir_2 = './datas/50div10gmm10/d2.csv'
    data_dir_3 = './datas/50div10gmm10/d3.csv'
    data_dir_all = './datas/50div10gmm10/dall.csv'
    dataset_1 = load_weighted_datasets(data_dir_1)
    dataset_2 = load_weighted_datasets(data_dir_2)
    dataset_3 = load_weighted_datasets(data_dir_3)
    dataset_all = load_weighted_datasets(data_dir_all)
    # LearnerEnv_all = LearnerEnv(        
    #     i_data=dataset_all,
    #     learning_steps = 200,
    #     learning_steps_max = 50000,
    #     feature_size = 50,)
    # for i in range(25):
    #     LearnerEnv_all.local_train(step=2000)
    #     LearnerEnv_all.test(dataset_all.sample(2000))
    penal = [[0, 1.5, 1.5], [1.5, 0, 1.5], [2.0, 2.0, 0]]
    BUFF_SIZE = 100
    buff_1 = {'i':dataset_2.sample(n=BUFF_SIZE), 'j':dataset_3.sample(n=BUFF_SIZE)}
    buff_2 = {'i':dataset_1.sample(n=BUFF_SIZE), 'j':dataset_3.sample(n=BUFF_SIZE)}
    buff_3 = {'i':dataset_1.sample(n=BUFF_SIZE), 'j':dataset_2.sample(n=BUFF_SIZE)}
    buff = {'1': buff_1, '2': buff_2, '3': buff_3}
    LearnerEnv_1 = LearnerEnv(        
        i_data=dataset_1,
        learning_steps = 200,
        learning_steps_max = 80000,
        feature_size = 50,)
    LearnerEnv_2 = LearnerEnv(        
        i_data=dataset_2,
        learning_steps = 200,
        learning_steps_max = 80000,
        feature_size = 50,)
    LearnerEnv_3 = LearnerEnv(        
        i_data=dataset_3,
        learning_steps = 200,
        learning_steps_max = 80000,
        feature_size = 50,)
    Env = {'1': LearnerEnv_1, '2': LearnerEnv_2, '3': LearnerEnv_3}
    with tf.variable_scope('DQN_with_prioritized_replay'):
        RL = DQNPrioritizedReplay(
            n_actions=5, #trans weights
            n_features=4,
            # learning_rate=0.01, #RL learning rate
            learning_rate=0.001, #retrain learning rate
            reward_decay=0.98, #reward decay --the infuluence of reward
            e_greedy=0.99,#max e-greedy
            e_greedy_increment=None, 
            replace_target_iter=20,
            memory_size=100,
            batch_size=10,

            output_graph=False,
            prioritized=True,
            dueling=False,
            sess=None, #for reload
        )    
    model_file = tf.train.latest_checkpoint('./ckpt/')
    tf.train.Saver().restore(RL.sess, model_file)
    observation1_1 = Env['1'].reset(buff['1']['i'], penal[0][1])
    observation1_2 = Env['1'].reset(buff['1']['j'], penal[0][2])
    observation2_1 = Env['2'].reset(buff['2']['i'], penal[1][0])
    observation2_2 = Env['2'].reset(buff['2']['j'], penal[1][2])
    observation3_1 = Env['3'].reset(buff['3']['i'], penal[2][0])
    observation3_2 = Env['3'].reset(buff['3']['j'], penal[2][1])
    step=0
    action = [0, 0, 0]
    while(Env['1'].done == False):
        observation1_1, observation1_2, buff['1']['i'], buff['1']['j'], action_num \
            = EnvStepOn(RL, Env['1'], Env['2'], Env['3'], observation1_1, observation1_2, \
                                buff['1']['i'], penal[0][1], buff['1']['j'], penal[0][2])
        action[0] += action_num
        observation2_1, observation2_2, buff['2']['i'], buff['2']['j'], action_num \
            = EnvStepOn(RL, Env['2'], Env['1'], Env['3'], observation2_1, observation2_2, \
                                buff['2']['i'], penal[1][0], buff['2']['j'], penal[1][2])
        action[1] += action_num
        observation3_1, observation3_2, buff['3']['i'], buff['3']['j'], action_num \
            = EnvStepOn(RL, Env['3'], Env['1'], Env['2'], observation3_1, observation3_2, \
                                buff['3']['i'], penal[2][0], buff['3']['j'], penal[2][1])
        action[2] += action_num
        if(step%10==0):
            print('action: ', action)
            data_tst = dataset_all.sample(n=2000)
            Env['1'].test(data_tst)
            Env['2'].test(data_tst)
            Env['3'].test(data_tst)
            acc = MajorVote(data_tst, Env['1'], Env['2'], Env['3'])
            print('accMV: ', acc)
        step+=1
        print('\n============================================\n')
    print('0')
    Env['1'].save_model('./ckpt/learners/8w-a1')
    Env['2'].save_model('./ckpt/learners/8w-a2')
    Env['3'].save_model('./ckpt/learners/8w-a3')
    acc = MajorVote(dataset_all, Env['1'], Env['2'], Env['3'])
    print('accMV: ', acc)

