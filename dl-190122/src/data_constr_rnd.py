import numpy as np
from numpy import *
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
from scipy.linalg import orth
import scipy.sparse


LABEL_NUM = 10

FEATURE_SIZE = 50
DIV = 10
GMM_COMPONENT = 10
GMM_COMPONENT_LOCAL = 10
GMM_NUM_EACH = 200
GMM_NUM_EACH_LOCAL = [20,400]
NODE_NUM=2
all_set = True
path = './datas/50div10gmm10'
def get_RndSymPosMatrix(size, divide, num):
    def get_A():
        D = np.diag(np.random.random_sample([size,]))/divide
        V = np.random.rand(size, size)
        U = orth(V)
        D = mat(D)
        U = mat(U)
        A = U.I * D * U
        return A
    return [get_A() for i in range(num)]

def get_RndMean(size, num):
    random_list = [np.random.random_sample([size,]) for i in range(num)]
    return random_list

# invariant_conv = [get_RndSymPosMatrix(size=FEATURE_SIZE) for i in range(GMM_COMPONENT)]

def data_construct(gmm_id, label, mean, conv, num):
    # x = np.empty(shape = [0,FEATURE_SIZE])
    # print('\nConstructing Gaussian Mixture Multivariate Dataof label -%d-:'%label)
    x = np.random.multivariate_normal(mean,conv,num)
    x = np.round(x, decimals=4)
    # random_append = np.random.normal(loc = 0.5,scale = 10,size = [num,APPEND_D])
    # x = np.concatenate((random_append, x), axis = 1)
    dataframe = pd.DataFrame(data = x)
    dataframe['label'] = label
    dataframe['gmm_id'] = gmm_id
    return dataframe

mean = [get_RndMean(size=FEATURE_SIZE, num =GMM_COMPONENT) for i in range(LABEL_NUM)]
conv = [get_RndSymPosMatrix(size=FEATURE_SIZE, divide=DIV, num=GMM_COMPONENT) for i in range(LABEL_NUM)]

ij_data_num = GMM_NUM_EACH
dataframe = pd.DataFrame()
for label in range(LABEL_NUM):
    mean_j = mean[label]
    conv_j = conv[label]
    for gmm_id in range(GMM_COMPONENT):
        mean_ij = mean_j[gmm_id]
        conv_ij = conv_j[gmm_id]
        dataframe_temp = data_construct(gmm_id=gmm_id, label=label, mean=mean_ij, conv=conv_ij, num=ij_data_num)
        dataframe = pd.concat((dataframe,dataframe_temp), axis=0, ignore_index=True)
# print(dataframe)
dataframe.to_csv(path+'-all'+ '.csv')



dataframe_1 = pd.DataFrame()
dataframe_2 = pd.DataFrame()
for label in range(LABEL_NUM):
    print('\nlabel: %d'  % (label,))
    num_list_1 = []
    num_list_2 = []
    mean_j = mean[label]
    conv_j = conv[label]
    rnd_list = random.sample(range(GMM_COMPONENT), GMM_COMPONENT_LOCAL)
    print(rnd_list)
    for gmm_id in rnd_list:
        mean_ij = mean_j[gmm_id]
        conv_ij = conv_j[gmm_id]
        if gmm_id > 4:
            ij_data_num_1 = np.random.randint(GMM_NUM_EACH_LOCAL[0]) #50instance each
            ij_data_num_2 = np.random.randint(GMM_NUM_EACH_LOCAL[1]) #200 instance each
            dataframe_temp_1 = data_construct(gmm_id=gmm_id, label=label, mean=mean_ij, conv=conv_ij, num=ij_data_num_1)
            dataframe_1 = pd.concat((dataframe_1,dataframe_temp_1), axis=0, ignore_index=True)
            dataframe_temp_2 = data_construct(gmm_id=gmm_id, label=label, mean=mean_ij, conv=conv_ij, num=ij_data_num_2)
            dataframe_2 = pd.concat((dataframe_2,dataframe_temp_2), axis=0, ignore_index=True)
            num_list_1.append(ij_data_num_1)
            num_list_2.append(ij_data_num_2)
        else:
            ij_data_num_1 = np.random.randint(GMM_NUM_EACH_LOCAL[1]) #50instance each
            ij_data_num_2 = np.random.randint(GMM_NUM_EACH_LOCAL[0]) #200 instance each
            dataframe_temp_1 = data_construct(gmm_id=gmm_id, label=label, mean=mean_ij, conv=conv_ij, num=ij_data_num_1)
            dataframe_1 = pd.concat((dataframe_1,dataframe_temp_1), axis=0, ignore_index=True)
            dataframe_temp_2 = data_construct(gmm_id=gmm_id, label=label, mean=mean_ij, conv=conv_ij, num=ij_data_num_2)
            dataframe_2 = pd.concat((dataframe_2,dataframe_temp_2), axis=0, ignore_index=True)
            num_list_1.append(ij_data_num_1)
            num_list_2.append(ij_data_num_2)
    print(num_list_1)
    print(num_list_2)
    # print(dataframe)
dataframe_1.to_csv(path+'-'+str(1)+ '.csv')
dataframe_2.to_csv(path+'-'+str(2)+ '.csv')
# dataframe = dataframe.sample(DATA_NUM)
# for i in range(LABEL_NUM):
#         dataframe_temp, _ = data_construct(label = i,gmm_id = 0, num = DATA_NUM, size = FEATURE_SIZE, gmm_size = GMM_COMPONENT)
#         dataframe = pd.concat((dataframe,dataframe_temp), axis=0, ignore_index=True)



# dataframe_A,_ = data_construct(label = 0, num = DATA_NUM)
# dataframe_B,_ = data_construct(label = 1, num = DATA_NUM)
# dataframe_C,_ = data_construct(label = 2, num = DATA_NUM)
# dataframe = pd.concat((dataframe_A,dataframe_B,dataframe_C),axis=0,ignore_index=True)
# dataframe.to_csv('./GMM_model/GMM_data/G_3M_4M_10000.csv')
# print(dataframe)
# print(dataframe)

# ## show 3D dots
# x = np.array(dataframe.loc[(dataframe.label==1),0])
# y = np.array(dataframe.loc[(dataframe.label==1),1])
# z = np.array(dataframe.loc[(dataframe.label==1),2])
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax.scatter(x, y, z, s=20, c='r', depthshade=True)
# x = np.array(dataframe.loc[(dataframe.label==2),0])
# y = np.array(dataframe.loc[(dataframe.label==2),1])
# z = np.array(dataframe.loc[(dataframe.label==2),2])
# ax.scatter(x, y, z, s=20, c='b', depthshade=True)
# x = np.array(dataframe.loc[(dataframe.label==0),0])
# y = np.array(dataframe.loc[(dataframe.label==0),1])
# z = np.array(dataframe.loc[(dataframe.label==0),2])
# ax.scatter(x, y, z, s=20, c='g', depthshade=True)
# x = np.array(dataframe.loc[(dataframe.label==3),0])
# y = np.array(dataframe.loc[(dataframe.label==3),1])
# z = np.array(dataframe.loc[(dataframe.label==3),2])
# ax.scatter(x, y, z, s=20, c='k', depthshade=True)
# x = np.array(dataframe.loc[(dataframe.label==4),0])
# y = np.array(dataframe.loc[(dataframe.label==4),1])
# z = np.array(dataframe.loc[(dataframe.label==4),2])
# ax.scatter(x, y, z, s=20, c='y', depthshade=True)
# x = np.array(dataframe.loc[(dataframe.label==5),0])
# y = np.array(dataframe.loc[(dataframe.label==5),1])
# z = np.array(dataframe.loc[(dataframe.label==5),2])
# ax.scatter(x, y, z, s=20, c='m', depthshade=True)
# x = np.array(dataframe.loc[(dataframe.label==6),0])
# y = np.array(dataframe.loc[(dataframe.label==6),1])
# z = np.array(dataframe.loc[(dataframe.label==6),2])
# ax.scatter(x, y, z, s=20, c='c', depthshade=True)
plt.show()
# ## show 3D dots
# x = np.array(dataframe.loc[(dataframe.label==1) & (dataframe.gmm_id==0),0])
# y = np.array(dataframe.loc[(dataframe.label==1) & (dataframe.gmm_id==0),1])
# z = np.array(dataframe.loc[(dataframe.label==1) & (dataframe.gmm_id==0),2])
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax.scatter(x, y, z, s=20, c='r', depthshade=True)
# x = np.array(dataframe.loc[(dataframe.label==2) & (dataframe.gmm_id==0),0])
# y = np.array(dataframe.loc[(dataframe.label==2) & (dataframe.gmm_id==0),1])
# z = np.array(dataframe.loc[(dataframe.label==2) & (dataframe.gmm_id==0),2])
# ax.scatter(x, y, z, s=20, c='b', depthshade=True)
# x = np.array(dataframe.loc[(dataframe.label==0) & (dataframe.gmm_id==0),0])
# y = np.array(dataframe.loc[(dataframe.label==0) & (dataframe.gmm_id==0),1])
# z = np.array(dataframe.loc[(dataframe.label==0) & (dataframe.gmm_id==0),2])
# ax.scatter(x, y, z, s=20, c='g', depthshade=True)
# x = np.array(dataframe.loc[(dataframe.label==3) & (dataframe.gmm_id==0),0])
# y = np.array(dataframe.loc[(dataframe.label==3) & (dataframe.gmm_id==0),1])
# z = np.array(dataframe.loc[(dataframe.label==3) & (dataframe.gmm_id==0),2])
# ax.scatter(x, y, z, s=20, c='k', depthshade=True)
# x = np.array(dataframe.loc[(dataframe.label==4) & (dataframe.gmm_id==0),0])
# y = np.array(dataframe.loc[(dataframe.label==4) & (dataframe.gmm_id==0),1])
# z = np.array(dataframe.loc[(dataframe.label==4) & (dataframe.gmm_id==0),2])
# ax.scatter(x, y, z, s=20, c='y', depthshade=True)
# x = np.array(dataframe.loc[(dataframe.label==5) & (dataframe.gmm_id==0),0])
# y = np.array(dataframe.loc[(dataframe.label==5) & (dataframe.gmm_id==0),1])
# z = np.array(dataframe.loc[(dataframe.label==5) & (dataframe.gmm_id==0),2])
# ax.scatter(x, y, z, s=20, c='m', depthshade=True)
# x = np.array(dataframe.loc[(dataframe.label==6) & (dataframe.gmm_id==0),0])
# y = np.array(dataframe.loc[(dataframe.label==6) & (dataframe.gmm_id==0),1])
# z = np.array(dataframe.loc[(dataframe.label==6) & (dataframe.gmm_id==0),2])
# ax.scatter(x, y, z, s=20, c='c', depthshade=True)
# plt.show()
