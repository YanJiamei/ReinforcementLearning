import tensorflow as tf
import functools
import numpy as np
import pandas as pd
from tool_func import load_datasets

class LearnerEnv:
    def __init__(
            self, 
            i_data,
            feature_size,
            label_size = 10,
            batch_size = 100,
            buffer_size = 100,
            hidden_layer_size = 100,
            learning_rate = 0.01,
            learning_steps = 100,
            learning_steps_max = 50000,    
            
    ):
        self.input_node = feature_size
        self.output_node = label_size
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.learning_rate = learning_rate
        self.n_l1 = hidden_layer_size

        # self.tst_data = tst_data
        # self.tst_num, _ = self.tst_data.shape
        self.i_data = self.local_data = i_data
        # self.j_data = self.trans_data = j_data
        self.buffer_data = pd.DataFrame()


        self.learn_step_counter = int(0)
        self.learning_steps = learning_steps
        self.learning_steps_max = learning_steps_max
        self.done = False

        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)
        self._build_net() #contains global initializer()
        
        self.sess.run(self.init_op)
        
    def _build_net(self):
        with self.graph.as_default():
            with tf.variable_scope('local_data'):
                self.x = tf.placeholder(tf.float32, [None, self.input_node], name = 'x-input')
                self.y_ = tf.placeholder(tf.int64, [None, ], name = 'y-input')
                self.w = tf.placeholder(tf.float32, [None, ], name = 'all-instance-weight-local')

            with tf.variable_scope('net'):
                
                #config of layers
                w_initializer, b_initializer = \
                    tf.truncated_normal_initializer(mean = 0., stddev = 0.1), tf.constant_initializer(value = 0.1)

                with tf.variable_scope('l1'):
                    w1 = tf.get_variable('w1', [self.input_node, self.n_l1], initializer=w_initializer)
                    b1 = tf.get_variable('b1', [self.n_l1], initializer=b_initializer)
                    l1 =  tf.nn.relu(tf.matmul(self.x, w1) + b1)
                    
                with tf.variable_scope('l2'):
                    w2 = tf.get_variable('w2', [self.n_l1, self.output_node], initializer=w_initializer)
                    b2 = tf.get_variable('b2', [self.output_node], initializer=b_initializer)
                    self.y = tf.matmul(l1, w2) + b2
                    self.soft_max_prob = tf.nn.softmax(self.y)
                with tf.variable_scope('weighted_loss'):
                    self.cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = self.y, labels = self.y_)
                    weighted_cross_entropy =tf.multiply(self.cross_entropy, self.w)
                    # use unweighted--loss to validation the reward, use weighted_loss to train
                    self.weighted_loss = tf.divide(tf.reduce_sum(weighted_cross_entropy), tf.reduce_sum(self.w))
                    self.loss = tf.reduce_mean(self.cross_entropy)
                with tf.variable_scope('train'):
                    self.global_step = tf.Variable(dtype=tf.int64, initial_value=0, trainable=True)
                    self.train = tf.train.AdagradOptimizer(self.learning_rate).minimize(self.weighted_loss, global_step=self.global_step)
                    
                with tf.variable_scope('accuracy'):
                    correct_prediction = tf.equal(tf.argmax(self.y, 1), self.y_) #axis=1
                    correct_prediction_float = tf.cast(correct_prediction, tf.float32)
                    self.accuracy = tf.reduce_mean(correct_prediction_float)
            self.init_op = tf.initialize_all_variables()
            self.saver = tf.train.Saver()

    def save_model(self, name):
        self.saver.save(self.sess, (name))

    def load_feed_dict(self,data,batch_size):
        batch = data.sample(n=batch_size)
        feed_dict = {
            self.x: batch.iloc[:,range(self.input_node)].as_matrix(), 
            self.y_: batch['label'].as_matrix(), 
            self.w: batch['weights'].as_matrix()
        }
        return feed_dict
        
    def get_softmax_output(self,data):
        feed_dict = {
            self.x: data.iloc[:,range(self.input_node)].as_matrix(), 
            self.y_: data['label'].as_matrix(), 
            self.w: data['weights'].as_matrix()
        }
        soft_max = self.sess.run(self.soft_max_prob, feed_dict=feed_dict)
        return soft_max

    def local_train(self, step):
        for i in range(step):
            _, loss, self.learn_step_counter= \
                self.sess.run([self.train, self.loss, self.global_step], feed_dict=self.load_feed_dict(data = self.local_data, batch_size = self.batch_size))
            if i%1000==0:
                print('loss: ',loss)
    def get_observation(self, buffdata, trans_penalty):
        self.buffer_data = buffdata
        self.H_ji = self.sess.run(self.loss, feed_dict=self.load_feed_dict(data = self.buffer_data, batch_size = self.buffer_size))
        self.H_ji_relative = (self.H_ji - self.H_ii_t) / self.H_ii_t #buffer数据对当前学习的重要性
        self.H_ii_relative = (self.H_ii_t - self.H_ii_0)/self.H_ii_0
        
        self.observation = np.array([self.H_ii_relative, self.H_ii_t, self.H_ji_relative, trans_penalty])
        return self.observation

    def step1_action(self, action, buffdata, trans_penalty, action_base = 1.0):
        self.H_ii_0 = self.H_ii_t
        if action==0:
            self.action_value = 0
        else:
            #concat Dji U Dii
            self.buffer_data = buffdata
            self.buffer_data['weights']= action_base * action
            self.local_data = pd.concat([self.local_data, self.buffer_data], axis=0, ignore_index=True)
            self.action_value = (self.H_ji - self.H_ii_0) - trans_penalty

    def step2_train_and_get_reward(self):
        past_loss = []
        for i in range(self.learning_steps):
            _, loss, self.learn_step_counter= \
                self.sess.run([self.train, self.loss, self.global_step], feed_dict=self.load_feed_dict(data = self.local_data, batch_size = self.batch_size))
            past_loss.append(loss)
        self.H_ii_t = np.mean(past_loss[-5:])
        self.state_value = -self.H_ii_t

        reward = self.action_value + self.state_value
        if self.learn_step_counter >= self.learning_steps_max:
            self.done = True
        else:
            self.done = False
        return reward
    def reset(self, buffdata, trans_penalty):
        #reset data and load buffer
        self.done = False
        self.local_data = self.i_data
        self.buffer_data = buffdata

        self.H_ii_0 = self.H_ii_t = self.sess.run(self.loss, feed_dict=self.load_feed_dict(data = self.local_data, batch_size = self.batch_size * 5))
        self.H_ii_relative = 1.
        self.H_ji = self.sess.run(self.loss, feed_dict=self.load_feed_dict(data = self.buffer_data, batch_size = self.buffer_size))
        self.H_ji_relative = (self.H_ji-self.H_ii_0)/self.H_ii_0
        
        self.observation =  np.array([self.H_ii_relative, self.H_ii_0, self.H_ji_relative, trans_penalty])
        return self.observation

    def test(self, dataset, batch_size=1000):
        loss, acc= self.sess.run([self.loss, self.accuracy] , feed_dict=self.load_feed_dict(data = dataset, batch_size = batch_size))
        print('\naccuracy on test set: ', acc)
        print('\nloss on test set: ', loss)
        return loss, acc

if __name__ == "__main__":
    from tool_func import *
    from PDDQN import *
    # log_dir = './logs'
    data_dir_1 = './datas/50div10gmm10/d1.csv'
    data_dir_2 = './datas/50div10gmm10/d2.csv'
    data_dir_3 = './datas/50div10gmm10/d3.csv'
    data_dir_all = './datas/50div10gmm10/dall.csv'
    dataset_1 = load_weighted_datasets(data_dir_1)
    dataset_2 = load_weighted_datasets(data_dir_2)
    dataset_3 = load_weighted_datasets(data_dir_3)
    dataset_all = load_weighted_datasets(data_dir_all)
    
    penal_i = 1.0
    penal_j = 1.0
    BUFF_SIZE = 100
    buff_i = dataset_2.sample(n=BUFF_SIZE)
    buff_j = dataset_3.sample(n=BUFF_SIZE)
    LearnerEnv_1 = LearnerEnv(        
        i_data=dataset_1,
        learning_steps = 200,
        learning_steps_max = 50000,
        feature_size = 50,)

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