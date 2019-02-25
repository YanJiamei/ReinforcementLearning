import tensorflow as tf
import functools
import numpy as np
import pandas as pd
from tool_func import load_datasets

class Learner_entro_reward:
    def __init__(
            self,
            # log_dir,    
            i_data,
            j_data,
            # tst_data,
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
        self.j_data = self.trans_data = j_data
        self.buffer_data = self.trans_data.sample(n=self.buffer_size)

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
                    # self.soft_max_prob = tf.nn.softmax(self.y)
                with tf.variable_scope('weighted_loss'):
                    self.cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = self.y, labels = self.y_)
                    weighted_cross_entropy =tf.multiply(self.cross_entropy, self.w)
                    self.loss = tf.divide(tf.reduce_sum(weighted_cross_entropy), tf.reduce_sum(self.w))
                    
                with tf.variable_scope('train'):
                    self.global_step = tf.Variable(dtype=tf.int64, initial_value=0, trainable=True)
                    self.train = tf.train.AdagradOptimizer(self.learning_rate).minimize(self.loss, global_step=self.global_step)
                    
                with tf.variable_scope('accuracy'):
                    correct_prediction = tf.equal(tf.argmax(self.y, 1), self.y_)
                    correct_prediction_float = tf.cast(correct_prediction, tf.float32)
                    self.accuracy = tf.reduce_mean(correct_prediction_float)

            self.init_op = tf.initialize_all_variables()

    def load_feed_dict(self,data,batch_size):
        batch = data.sample(n=batch_size)
        feed_dict = {
            self.x: batch.iloc[:,range(self.input_node)].as_matrix(), 
            self.y_: batch['label'].as_matrix(), 
            self.w: batch['weights'].as_matrix()
        }
        return feed_dict
    # def reweight_data(self, data, action, base = 1.0):
    #     data['weights']=base * action
    #     return data


        # self.local_data = pd.concat([self.local_data, self.buffer_data], axis=0, ignore_index=True)
        # self.buffer_data = self.trans_data.sample(n=self.buffer_size)
        # self.buffer_data['weights']=1.0

    def step(self, action, trans_penalty, action_base = 1.0):
        entro_gap = (self.H_ji - self.H_ii_0)
        past_loss = []
        if action==0:
            action_flag = 0
            for i in range(self.learning_steps):
                _, loss, self.learn_step_counter= \
                    self.sess.run([self.train, self.loss, self.global_step], feed_dict=self.load_feed_dict(data = self.local_data, batch_size = self.batch_size))
                past_loss.append(loss)
        else:
            action_flag = 1
            #action>0: reweight buffer data
            self.buffer_data['weights']= action_base * action
            #train buffer first
            # for i in range(int(self.learning_steps*0.1)):
            #     _, loss, self.learn_step_counter= \
            #         self.sess.run([self.train, self.loss, self.global_step], feed_dict=self.load_feed_dict(data = self.buffer_data, batch_size = self.buffer_size))
            #     past_loss.append(loss)
            #concat Dji U Dii
            self.local_data = pd.concat([self.local_data, self.buffer_data], axis=0, ignore_index=True)
        
            #local training
            for i in range(int(self.learning_steps*1.0)):
                _, loss, self.learn_step_counter= \
                    self.sess.run([self.train, self.loss, self.global_step], feed_dict=self.load_feed_dict(data = self.local_data, batch_size = self.batch_size))
                past_loss.append(loss)
            #reload buffer
            self.buffer_data = self.trans_data.sample(n=self.buffer_size)
            self.buffer_data['weights']=1.0

        #observation Hji - Hii
         #F1 AVE LOSS
        buffer_partition = self.buffer_size / self.local_data.shape[0]
        iter_partion = self.learn_step_counter / self.learning_steps_max
        self.H_ii_t = np.mean(past_loss[-5:])
        
        acc_ii = self.sess.run(self.accuracy, feed_dict=self.load_feed_dict(data = self.local_data, batch_size = self.batch_size * 5))
        self.H_ji, acc_ji = self.sess.run([self.loss, self.accuracy], feed_dict=self.load_feed_dict(data = self.buffer_data, batch_size = self.buffer_size))
        
        reward = entro_gap * action_flag - trans_penalty * action_flag - self.H_ii_t
        self.H_ii_0 = self.H_ii_t
        observation =  np.array([buffer_partition, iter_partion, self.H_ii_0, acc_ii, self.H_ji, acc_ji])
        

        if self.learn_step_counter >= self.learning_steps_max:
            self.done = True
        else:
            self.done = False
        
        return observation, reward, self.done

    def reset(self):
        #reset data and load buffer
        self.local_data = self.i_data
        self.trans_data = self.j_data
        self.buffer_data = self.trans_data.sample(n=self.buffer_size)
        #init vals
        self.learn_step_counter = 0
        self.done = False
        self.sess.run(self.init_op)

        buffer_partition = self.buffer_size / self.local_data.shape[0]
        iter_partion = 0
        self.H_ii_0, acc_ii = self.sess.run([self.loss, self.accuracy], feed_dict=self.load_feed_dict(data = self.local_data, batch_size = self.batch_size * 5))
        self.H_ji, acc_ji = self.sess.run([self.loss, self.accuracy], feed_dict=self.load_feed_dict(data = self.buffer_data, batch_size = self.buffer_size))

        observation =  np.array([buffer_partition, iter_partion, self.H_ii_0, acc_ii, self.H_ji, acc_ji])
        return observation
    def test(self, dataset, batch_size=1000):
        loss, acc= self.sess.run([self.loss, self.accuracy] , feed_dict=self.load_feed_dict(data = dataset, batch_size = batch_size))
        print('\naccuracy on test set: ', acc)
        print('\nloss on test set: ', loss)
        return loss, acc

if __name__ == "__main__":
    from tool_func import *
    log_dir = './logs'
    data_dir = './datas/50div10gmm10-1.csv'
    dataset_0 = load_datasets(data_dir)
    env = Learner(
        log_dir=log_dir,
        i_data=dataset_0['train'],
        j_data=dataset_0['train'],
        feature_size = 50,
    )
    k=env.reset()
    # print(env.reset())

    action = 2
    b=env.step(action=action, trans_penalty = 0.1)

    k1 = env.reset()
    k2 = env.step(action=action, trans_penalty = 0.1)
    print('00000000000000')
    # env.reset()
    # env.step(action=0)
    # env.step(action=0)
