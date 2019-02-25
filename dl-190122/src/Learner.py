"""
Yan Jiamei
"""
import tensorflow as tf
import functools
import numpy as np
import pandas as pd
from tool_func import load_datasets
# tf.set_random_seed(1)

INPUT_NODE = 50
TRANING_STEPS = 1000
TRANING_STEPS_MAX = 50000
class Learner:
    def __init__(
            self,
            log_dir,    
            i_data,
            j_data,
            tst_data,
            feature_size = INPUT_NODE,
            label_size = 10,
            batch_size = 100,
            buffer_size = 100,
            hidden_layer_size = 100,
            learning_rate = 0.01,
            learning_steps = TRANING_STEPS,
            learning_steps_max = TRANING_STEPS_MAX,    
            
    ):
        
        self.input_node = feature_size
        self.output_node = label_size
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.learning_rate = learning_rate
        self.n_l1 = hidden_layer_size

        self.reward_tst_data = tst_data
        self.tst_num, _ = tst_data.shape
        self.i_data = self.local_data = i_data
        self.j_data = self.trans_data = j_data
        self.buffer_data = self.trans_data.sample(n=self.buffer_size)

        self.learn_step_counter = int(0)
        self.learning_steps = learning_steps
        self.learning_steps_max = learning_steps_max
        self.done = False
        self.loss_list = [[],[]]
        
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)
        self._build_net()
        self.writer = tf.summary.FileWriter(logdir = log_dir, graph=self.graph)
        self.sess.run(self.init_op)
        # self.saver = tf.train.Saver()
        # self.saver.save(self.sess,'./initial_learner_vars')
    def _build_net(self):
        with self.graph.as_default():
            with tf.variable_scope('local_data'):
                self.x = tf.placeholder(tf.float32, [None, self.input_node], name = 'x-input')
                self.y_ = tf.placeholder(tf.int64, [None, ], name = 'y-input')
                self.w = tf.placeholder(tf.float32, [None, ], name = 'all-instance-weight-local')

            # with tf.variable_scope('received_data'):
            #     self.x_trans = tf.placeholder(tf.float32, [None, self.input_node], name = 'x-input')
            #     self.y_trans = tf.placeholder(tf.int64, [None, ], name = 'y-input')
            #     self.w_trans = tf.placeholder(tf.float32, [None, ], name = 'instance-weight-trans')
            with tf.variable_scope('net'):
                
                #config of layers
                w_initializer, b_initializer = \
                    tf.truncated_normal_initializer(mean = 0., stddev = 0.1), tf.constant_initializer(value = 0.1)
                with tf.variable_scope('l1'):
                    w1 = tf.get_variable('w1', [self.input_node, self.n_l1], initializer=w_initializer)
                    b1 = tf.get_variable('b1', [self.n_l1], initializer=b_initializer)
                    l1 =  tf.nn.relu(tf.matmul(self.x, w1) + b1)
                    # l1_trans = tf.nn.relu(tf.matmul(self.x_trans, w1) + b1)
                with tf.variable_scope('l2'):
                    w2 = tf.get_variable('w2', [self.n_l1, self.output_node], initializer=w_initializer)
                    b2 = tf.get_variable('b2', [self.output_node], initializer=b_initializer)
                    self.y = tf.matmul(l1, w2) + b2
                    # self.inference_trans = tf.matmul(l1_trans, w2) + b2
                # with tf.variable_scope('local_infer'):   
                #     self.y = self.inference
                # with tf.variable_scope('trans_infer'):
                #     self.y_t = self.inference_trans

                with tf.variable_scope('weighted_loss'):
                    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = self.y, labels = self.y_)
                    weighted_cross_entropy =tf.multiply(cross_entropy, self.w)
                    self.loss = tf.divide(tf.reduce_sum(weighted_cross_entropy), tf.reduce_sum(self.w))
                    self.scalar_loss = tf.summary.scalar('wce_loss', self.loss)
                with tf.variable_scope('train'):
                    self.global_step = tf.Variable(dtype=tf.int64, initial_value=0, trainable=True)
                    self.train = tf.train.AdagradOptimizer(self.learning_rate).minimize(self.loss, global_step=self.global_step)
                    
                with tf.variable_scope('accuracy'):
                    correct_prediction = tf.equal(tf.argmax(self.y, 1), self.y_)
                    correct_prediction_float = tf.cast(correct_prediction, tf.float32)
                    self.accuracy = tf.reduce_mean(correct_prediction_float)
                    self.scalar_accuracy = tf.summary.scalar('accuracy', self.accuracy)
                    self.test_accuracy = tf.summary.scalar('test_accuracy', self.accuracy)
            self.init_op = tf.initialize_all_variables()
    
    def reload_buffer(self, action, base = 1.0):
        self.buffer_data['weights']=base * action
        self.local_data = pd.concat([self.local_data, self.buffer_data], axis=0, ignore_index=True)
        self.buffer_data = self.trans_data.sample(n=self.buffer_size)
        self.buffer_data['weights']=1.0

    def load_feed_dict(self,data,batch_size):
        batch = data.sample(n=batch_size)
        feed_dict = {
            self.x: batch.iloc[:,range(self.input_node)].as_matrix(), 
            self.y_: batch['label'].as_matrix(), 
            self.w: batch['weights'].as_matrix()
        }
        return feed_dict

    # def get_observation(self):

    def step(self, action, sigma = 0.2, tst_batch = 500):       
        #calculate rewards
        if action == 0:
            flag = 0.
        else:
            flag = 1.
            self.reload_buffer(action)

        
        #training in learning steps after action
        for i in range(self.learning_steps):
            # _, self.learn_step_counter, loss_, scalar_loss = \
            #     self.sess.run([self.train, self.global_step, self.loss, self.scalar_loss], feed_dict=self.load_feed_dict(data = self.local_data, batch_size = self.batch_size))
            # self.writer.add_summary(scalar_loss, self.learn_step_counter)
            _, self.learn_step_counter= \
                self.sess.run([self.train, self.global_step], feed_dict=self.load_feed_dict(data = self.local_data, batch_size = self.batch_size))
            
        #add reward, update observation
        local_acc, local_loss = self.sess.run([self.accuracy, self.loss], feed_dict=self.load_feed_dict(data = self.local_data, batch_size = self.batch_size))
        trans_acc, trans_loss = self.sess.run([self.accuracy, self.loss], feed_dict=self.load_feed_dict(data=self.buffer_data, batch_size=self.buffer_size))
        # reward = reward + local_acc
        buffer_partition = np.divide(self.buffer_size, self.local_data.shape[0], dtype = np.float32)
        observation = np.array([local_acc, local_loss, trans_acc, trans_loss, buffer_partition])
        tst_acc, tst_loss = self.sess.run([self.accuracy, self.loss], feed_dict=self.load_feed_dict(data = self.reward_tst_data, batch_size = self.tst_num))
        self.tst_loss_t = tst_loss
        reward = (self.tst_loss_0 - self.tst_loss_t) - flag * sigma
        self.tst_loss_0 = self.tst_loss_t
        self.loss_list[0].append(local_loss)
        self.loss_list[1].append(tst_loss)
        if self.learn_step_counter >= self.learning_steps_max:
            self.done = True
        else:
            self.done = False
        return observation, reward, self.done, self.loss_list
        

    def reset(self):
        self.local_data = self.i_data
        self.trans_data = self.j_data
        self.buffer_data = self.trans_data.sample(n=self.buffer_size)
        self.loss_list = [[],[]]

        self.learn_step_counter = 0
        self.done = False
        self.sess.run(self.init_op)
        # self.saver.restore(self.sess, save_path = './initial_learner_vars')
        local_acc, local_loss = self.sess.run([self.accuracy, self.loss], feed_dict=self.load_feed_dict(data = self.local_data, batch_size = self.batch_size))
        trans_acc, trans_loss = self.sess.run([self.accuracy, self.loss], feed_dict=self.load_feed_dict(data=self.buffer_data, batch_size=self.buffer_size))
        buffer_partition = np.divide(self.buffer_size, self.local_data.shape[0], dtype = np.float32)
        observation = np.array([local_acc, local_loss, trans_acc, trans_loss, buffer_partition])
        self.tst_loss_0 = local_loss
        return observation
    
    def test(self, batch_size=100):
        tst_loss, total_acc, scalar_acc = self.sess.run([self.loss, self.accuracy, self.test_accuracy] , feed_dict=self.load_feed_dict(data = self.reward_tst_data, batch_size = batch_size))
        self.writer.add_summary(scalar_acc, self.learn_step_counter)
        print('\naccuracy on test set: ', total_acc)
        print('\nloss on test set: ', tst_loss)
        return tst_loss, total_acc
    
    def destroy(self):
        self.writer.close()
        self.sess.close()

if __name__ == "__main__":
    log_dir = './5.1_Double_DQN/logs'
    data_dir = './5.1_Double_DQN/datas/origin50div50gmm10.csv'
    train_data, test_data, dataset_0, dataset_1, dataset_2 = load_datasets(data_dir)
    env = Learner(
        log_dir=log_dir,
        i_data=dataset_0,
        j_data=dataset_1
    )
    env.reset()
    for i in range(5):
        env.step(action=0)
        env.step(action=4)
    print('00000000000000')
    env.reset()
    env.step(action=0)
    env.step(action=0)
    