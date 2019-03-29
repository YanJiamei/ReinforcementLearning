import LearnerEnv 
from tool_func import *
from PDDQN import *
if __name__ == "__main__":
    data_dir_1 = './datas/50div10gmm10/d1.csv'
    data_dir_2 = './datas/50div10gmm10/d2.csv'
    data_dir_3 = './datas/50div10gmm10/d3.csv'
    data_dir_all = './datas/50div10gmm10/dall.csv'
    dataset_1 = load_weighted_datasets(data_dir_1)
    dataset_2 = load_weighted_datasets(data_dir_2)
    dataset_3 = load_weighted_datasets(data_dir_3)
    dataset_all = load_weighted_datasets(data_dir_all)
    
    penal = [[0, 1.5, 1.5], [1.5, 0, 1.5], [1.5, 1.5, 0]]
    BUFF_SIZE = 100
    buff_1 = {'i':dataset_2.sample(n=BUFF_SIZE), 'j':dataset_3.sample(n=BUFF_SIZE)}
    buff_2 = {'i':dataset_1.sample(n=BUFF_SIZE), 'j':dataset_3.sample(n=BUFF_SIZE)}
    buff_3 = {'i':dataset_1.sample(n=BUFF_SIZE), 'j':dataset_2.sample(n=BUFF_SIZE)}
    buff = {'1': buff_1, '2': buff_2, '3': buff_3}
    LearnerEnv_1 = LearnerEnv(        
        i_data=dataset_1,
        learning_steps = 200,
        learning_steps_max = 50000,
        feature_size = 50,)
    LearnerEnv_2 = LearnerEnv(        
        i_data=dataset_2,
        learning_steps = 200,
        learning_steps_max = 50000,
        feature_size = 50,)
    LearnerEnv_3 = LearnerEnv(        
        i_data=dataset_3,
        learning_steps = 200,
        learning_steps_max = 50000,
        feature_size = 50,)
    Env = {'1': LearnerEnv_1, '2': LearnerEnv_2, '3': LearnerEnv_3}
    with tf.variable_scope('DQN_with_prioritized_replay'):
        RL = DQNPrioritizedReplay(
            n_actions=5, #trans weights
            n_features=4,
            # learning_rate=0.01, #RL learning rate
            learning_rate=0.001, #retrain learning rate
            reward_decay=0.98, #reward decay --the infuluence of reward
            e_greedy=0.98,#max e-greedy
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

    while(env1.done == False):
        env1.reset(buff['1']['1'],penal[0][0])
        env1.reset(buff['1']['2'],penal[0][1])
        env1.action_and_step(observation1,observation2)
        

    observation_i = LearnerEnv_1.reset(buff_i, penal_i)
    observation_j = LearnerEnv_1.reset(buff_j, penal_j)
    step = 0
    action_num = 0
    while(LearnerEnv_1.done == False):
        
        print('obs.:')
        print('i:', observation_i)
        print('j:', observation_j)
        action_i, q_i_ , _i = RL.choose_action(observation_i)
        q_i = q_i_[0][action_i]
        print('action_1: ', action_i, '\tq_i: ', q_i, _i)
        action_j, q_j_ , _j = RL.choose_action(observation_j)
        q_j = q_j_[0][action_j]
        print('action_2: ', action_j, '\tq_j: ', q_j, _j)
        if(action_i>0 and action_j>0):
            action_num+=1
            if(q_i>q_j):
                print('choosing iiiii...\n')
                LearnerEnv_1.step1_action(action_i, buff_i, penal_i)
                reward = LearnerEnv_1.step2_train_and_get_reward()
                print('reward:', reward)
                buff_i = dataset_2.sample(n=BUFF_SIZE)
            else:
                print('choosing jjj...\n')
                LearnerEnv_1.step1_action(action_j, buff_j, penal_j)
                reward = LearnerEnv_1.step2_train_and_get_reward()
                print('reward:', reward,'\n')
                buff_j = dataset_3.sample(n=BUFF_SIZE)
        elif(action_i!=0):
            action_num+=1
            print('choosing iiiii...\n')
            LearnerEnv_1.step1_action(action_i, buff_i, penal_i)
            reward = LearnerEnv_1.step2_train_and_get_reward()
            print('reward:', reward)
            buff_i = dataset_2.sample(n=BUFF_SIZE)
        elif(action_j!=0):
            action_num+=1
            print('choosing jjj...\n')
            LearnerEnv_1.step1_action(action_j, buff_j, penal_j)
            reward = LearnerEnv_1.step2_train_and_get_reward()
            print('reward:', reward,'\n')
            buff_j = dataset_3.sample(n=BUFF_SIZE)
        else:
            print('no trans...')
            LearnerEnv_1.step1_action(0, buff_j, penal_j)
            reward = LearnerEnv_1.step2_train_and_get_reward()
            print('reward:', reward,'\n')
        if(step%5==0):
            data_tst = dataset_all.sample(n=2000)
            LearnerEnv_1.test(data_tst)
            

        observation_i = LearnerEnv_1.get_observation(buff_i, penal_i)
        observation_j = LearnerEnv_1.get_observation(buff_j, penal_j)
        step+=1

    print('action_num:', action_num)