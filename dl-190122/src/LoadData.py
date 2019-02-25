import pandas as pd
import tensorflow as tf
import numpy as np
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
print(dir_path)
tf.logging.set_verbosity(tf.logging.INFO)
models_path = os.path.join(dir_path, './logs/tst37/')
checkpoint = os.path.join(models_path, 'checkpoint')

# dataframe = pd.read_csv('./datas/100000/origin50div50gmm10.csv', index_col=0)
# dataframe = dataframe.sample(frac=1)
# train_data = dataframe[0:100000]
# test_data = dataframe[100000:110000]
FEATURE_NUM = 50
CLASSES = 10
dataframe = pd.read_csv('./datas/50div10gmm10-1.csv',index_col= 0 ,dtype=np.float32)
# dataframe = dataframe.sample(frac=1)
dataframe[['label','gmm_id']] = dataframe[['label','gmm_id']].astype(np.int64)
print(dataframe)
dataframe = dataframe.reset_index(drop = True)
data_num, _ = dataframe.shape
dataframe['weights'] = 1.0
dataframe['weights'] = dataframe['weights'].astype(np.float32)
print(dataframe)
train_num = int(data_num*0.9)
# print(dataframe)
dataframe = dataframe.sample(frac=1)
train_data = dataframe[0:train_num]
test_data = dataframe[train_num:data_num]
# print(train_data[0:20],test_data[0:20])
feature_list = [(str(x), np.array(train_data[str(x)])) for x in range(FEATURE_NUM)]
feature_list = dict(feature_list)
dataset = tf.data.Dataset.from_tensor_slices(
    (
        feature_list,
        np.array(train_data['label'].astype(int))
    )
)
# dataset = dataset.repeat(10).batch(100)
# train_data = dataset.range()
def train_input_fn(dataset, batch_size):
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)
    return dataset.make_one_shot_iterator().get_next()
# a = train_input_fn(dataset=dataset,batch_size=100)
my_feature_columns = []
for key,_ in feature_list.items():
    my_feature_columns.append(tf.feature_column.numeric_column(key = key))

# config = tf.ConfigProto(intra_op_parallelism_threads = 0,inter_op_parallelism_threads = 0)

# run_config = tf.ConfigProto()

classifier = tf.estimator.DNNClassifier(model_dir=models_path,batch_norm=False, feature_columns=my_feature_columns, n_classes=CLASSES, hidden_units = [100])
classifier.train(input_fn =lambda: train_input_fn(dataset=dataset, batch_size=100), steps=5000)
# classifier = tf.estimator.DNNClassifier(model_dir=)
feature_list_test = [(str(x), np.array(test_data[str(x)])) for x in range(FEATURE_NUM)]
feature_list_test = dict(feature_list_test)
dataset_test = tf.data.Dataset.from_tensor_slices(
    (
        feature_list_test,
        np.array(test_data['label'].astype(int))
    )
)
# dataset_test = dataset_test.batch(100)
# train_data = dataset.range()
def test_input_fn(dataset, batch_size):
    dataset = dataset.batch(batch_size)
    return dataset.make_one_shot_iterator().get_next()
# input_e = dataset.shuffle(20000).batch(100)
# eval_result = classifier.evaluate
eval_result = classifier.evaluate(input_fn =lambda: test_input_fn(dataset=dataset_test, batch_size=100),steps=1000)
print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))