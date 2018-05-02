import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from numpy.linalg import inv
if __name__ == '__main__':
    dataframe = pd.read_csv("dataset/train_all.csv")
    dataframe_test = pd.read_csv("dataset/test_all.csv", index_col='id')
    sns.distplot(dataframe['R'])
    plt.show()
    sns.distplot(dataframe['G'])
    plt.show()
    sns.distplot(dataframe['B'])
    plt.show()
    cov1 = dataframe.iloc[:, 1:3].cov()
    sns.heatmap(cov1)
    plt.show()
    cov2 = dataframe.iloc[:, 2:4].cov()
    sns.heatmap(cov2)
    plt.show()
    cov3 = dataframe.iloc[:, [1,3]].cov()
    sns.heatmap(cov3)
    plt.show()
    cov4 = dataframe.iloc[:, 1:4].cov()
    sns.heatmap(cov4)
    plt.show()

    # Xrandom
    class_1_X = dataframe.iloc[:40687, 1:4].values
    class_2_X = dataframe.iloc[40687:, 1:4].values

    #coveriance of class 1 &2
    cov1 = dataframe.iloc[:40687, 1:4].cov()
    cov2 = dataframe.iloc[40687:, 1:4].cov()
    #coveriance inverse of class1 &2
    cov1_inverse=inv(cov1)
    cov2_inverse=inv(cov2)
    print(cov1_inverse,cov2_inverse)
     #mean of class 1  & 2 and its tranpose
    class_1_mean = [dataframe.iloc[:40687, 1].values.mean(),dataframe.iloc[:40687, 2].values.mean(),dataframe.iloc[:40687, 3].values.mean(),]
    class_1_mean_inverse = np.transpose(class_1_mean)
    class_2_mean = [dataframe.iloc[40687:, 1].values.mean(),dataframe.iloc[40687:, 2].values.mean(),dataframe.iloc[40687:, 3].values.mean()]
    class_2_mean_inverse = np.transpose(class_2_mean)
    print(class_1_mean_inverse,class_2_mean_inverse)
    #ln sigma 1 / sigma 2
    print(np.log(cov1/cov2))
    print(cov1)
    print(cov2)
    x=dataframe_test.iloc[1,:]
    x_transpose=np.transpose(x)
    Quadritic_term= x_transpose*(cov2_inverse-cov1_inverse)*x
   # print(qda(class_1_X,class_1_mean,cov1,1))

