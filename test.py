import pandas as pd
from random import shuffle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import mean_squared_error
import torch

def Creat_new_two_dimensional_rating(train_data):
    #print(train_data.columns)     #查看dateframe里面的数据的列名，方便后续的处理
    # 透视数据集，生成以人（userId）和电影（movieId）为索引的二维数组
    train_data_df = pd.DataFrame(train_data, columns=['userId', 'movieId', 'rating', 'timestamp'])
    pivot_data = train_data_df.pivot_table(index='userId', columns='movieId', values='rating')
    #绘制图形
    # 获取x轴和y轴的长度
    x = pivot_data.columns.values
    y = pivot_data.index.values
    z = pivot_data.values
    # 创建3D图像
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # 绘制3D图像
    x, y = np.meshgrid(x, y)
    surf = ax.plot_surface(x, y, z, cmap='viridis')
    # 显示图形
    plt.show()
    #print(pivot_data)
    #将文件存储到特定的位置
    #excel_file = 'archive/new_two_dimensional_ratings.csv'
    # 将数据保存到Excel文件中
    #pivot_data.to_csv(excel_file, index=True)

def data_exploration():
    # 读入训练集
    ifname = r"archive\ratings.csv"
    ifname2 = r"archive\movies.csv"
    ifname3 = r"archive\tags.csv"
    movie_data = pd.read_csv(ifname2)
    tags_data = pd.read_csv(ifname3)
    data = pd.read_csv(ifname)
    # 删除时间列
    data = data.drop('timestamp', axis=1)
    print(movie_data.columns)
    print(tags_data.columns)
    print(data.columns)
    # 合并数据，左连接方式
    merged_data = pd.merge(data, tags_data, on=['userId', 'movieId'], how='left')
    print(merged_data.info())
    print(merged_data.describe())
    print("缺失值的个数")
    print(merged_data['tag'].isnull().sum())
    print()
    print()
    merged_data_second = pd.merge(merged_data, movie_data, on='movieId', how='left')
    merged_data_second.drop('title', axis=1, inplace=True)
    merged_data_second.drop('timestamp', axis=1, inplace=True)
    print(merged_data_second.info())
    print(merged_data_second.describe())
    print("缺失值的个数")
    print(merged_data_second['genres'].isnull().sum())
    print("\n" + "数据集合里面的 tag 标签的所有可能输出")
    listt = merged_data_second['tag'].unique()
    print(listt.shape)
    for value in listt:
        print(value)

if __name__ == "__main__":
    ifname = r"archive\ratings.csv"
    ifname2 = r"archive\movies.csv"
    ifname3 = r"archive\tags.csv"
    movie_data = pd.read_csv(ifname2)
    tags_data = pd.read_csv(ifname3)
    data = pd.read_csv(ifname)
    data = data.drop('timestamp', axis=1)
    merged_data = pd.merge(data, tags_data, on=['userId', 'movieId'], how='left')
    merged_data_second = pd.merge(merged_data, movie_data, on='movieId', how='left')
    merged_data_second.drop('title', axis=1, inplace=True)
    merged_data_second.drop('timestamp', axis=1, inplace=True)
    excel_file = 'archive/merged_data_second.csv'
    # 将数据保存到Excel文件中
    #merged_data_second.to_csv(excel_file, index=True)
    print(merged_data_second.columns)
    print(merged_data_second['tag'].isnull().sum())
    '''data = merged_data_second
    X_text = data[['genres', 'tag']]
    X_numeric = data[['userId', 'movieId']]
    y_mean = data['rating'].mean()
    # 将最终评分转换为二分类标签

        # 将电影类型和评价转换为词袋特征
    vectorizer = CountVectorizer()
    X_text = X_text.apply(lambda x: ' '.join(map(str, x)), axis=1)
    X_text = vectorizer.fit_transform(X_text)

        # 合并文本特征和数值特征
    X = np.concatenate((X_text.toarray(), X_numeric.values), axis=1)

    y = np.where(data['rating'] >= y_mean, 1, 0)

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=58)

    # 模型训练
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)

    # 模型预测
    y_pred = regressor.predict(X_test)
    print(1,y_pred[:20])
    print(2,y_test[:20])
    # 定义阈值，将连续值预测结果转换为二分类标签
    threshold = 0.5
    y_pred_binary = np.where(y_pred >= threshold, 1, 0)

    # 计算预测结果的成功率
    accuracy = accuracy_score(y_test, y_pred_binary)
    print(y_mean," Accuracy: ", accuracy)'''
    #data_exploration()
    '''print(movie_data)
    print(tags_data)
    data = data.values
    #print(data)
    print(data.shape)
    #随机化数据
    shuffle(data)
    #取前80作为训练集
    p = 0.8
    train_data = data[:int(p * len(data)),:]
    print(train_data.shape)
    test_data = data[int(p * len(data)):,:]
    Creat_new_two_dimensional_rating(train_data)'''
