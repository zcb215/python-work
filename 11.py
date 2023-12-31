import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from matplotlib import pyplot as plt
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
from torch.optim.lr_scheduler import StepLR

# 读取数据
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
data = merged_data_second
# 特征选择
data['tag'].fillna('', inplace=True)
text_genres_features = data['genres']
text_tag_features = data['tag']
numeric_userId_features = data['userId']
numeric_movieId_features = data['movieId']
target = data['rating']

#print(data['rating'].mean())    #3.514813444101405

#将分数集合  /  10进入计算
target = torch.Tensor(target.values).float().unsqueeze(1)
#target = torch.div(target, 10)

# 将文本特征转换为词袋特征
vectorizer = CountVectorizer(token_pattern=r'\b\w+\b|\|')
text_tag_features = text_tag_features.fillna('')
text_genres_features = vectorizer.fit_transform(text_genres_features).toarray()
text_tag_features = vectorizer.fit_transform(text_tag_features).toarray()
#print(text_tag_features.shape)   # (102677, 1720)
#print(text_tag_features)
#print(text_genres_features.shape)   # (102677, 25)
#print(text_genres_features)

# 对数值型特征进行最小-最大缩放归一化
scaler = MinMaxScaler()
numeric_userId_features = scaler.fit_transform(numeric_userId_features.values.reshape(-1, 1))
numeric_movieId_features = scaler.fit_transform(numeric_movieId_features.values.reshape(-1, 1))
text_genres_features = scaler.fit_transform(text_genres_features)
text_tag_features = scaler.fit_transform(text_tag_features)
#print(numeric_userId_features)
#print(numeric_movieId_features)
#print(text_genres_features)
#print(text_tag_features)

# 将特征转换为Tensor
#text_features = torch.Tensor(text_features)
#numeric_features = torch.Tensor(numeric_features.values)
#numeric_features = torch.Tensor(numeric_features.values)
numeric_userId_features = torch.Tensor(numeric_userId_features)
numeric_movieId_features = torch.Tensor(numeric_movieId_features)
text_genres_features = torch.Tensor(text_genres_features)
text_tag_features = torch.Tensor(text_tag_features)

# 合并特征
features = torch.cat((numeric_userId_features,numeric_movieId_features,text_genres_features,text_tag_features), dim=1)
#print(features.shape)   #torch.Size([102677, 1747])
#
# 划分训练集和测试集

train_features, test_features, train_target, test_target = train_test_split(
    features, target, test_size=0.2, random_state=42
)
if torch.cuda.is_available():
    print("GPU is available")
    device = torch.device('cuda')
else:
    print("GPU is not available")
    device = torch.device('cpu')
# 划分区间
numeric_userId_dim = numeric_userId_features.shape[1]
numeric_movieId_dim = numeric_movieId_features.shape[1]
text_genres_dim = text_genres_features.shape[1]
text_tag_dim = text_tag_features.shape[1]

train_numeric_userId_features = train_features[:, :numeric_userId_dim]
train_numeric_movieId_features = train_features[:, numeric_userId_dim:numeric_movieId_dim + numeric_userId_dim]
train_text_genres_features = train_features[:, numeric_movieId_dim + numeric_userId_dim:numeric_movieId_dim + numeric_userId_dim + text_genres_dim]
train_text_tag_features = train_features[:, numeric_movieId_dim + numeric_userId_dim + text_genres_dim:numeric_movieId_dim + numeric_userId_dim + text_genres_dim + text_tag_dim]

train_numeric_userId_features.to(device)
train_numeric_movieId_features.to(device)
train_text_genres_features.to(device)
train_text_tag_features.to(device)

test_numeric_userId_features = test_features[:, :numeric_userId_dim]
test_numeric_movieId_features = test_features[:, numeric_userId_dim:numeric_movieId_dim + numeric_userId_dim]
test_text_genres_features = test_features[:, numeric_movieId_dim + numeric_userId_dim:numeric_movieId_dim + numeric_userId_dim + text_genres_dim]
test_text_tag_features = test_features[:, numeric_movieId_dim + numeric_userId_dim + text_genres_dim:numeric_movieId_dim + numeric_userId_dim + text_genres_dim + text_tag_dim]

test_numeric_userId_features.to(device)
test_numeric_movieId_features.to(device)
test_text_genres_features.to(device)
test_text_tag_features.to(device)

#print(train_numeric_userId_features.shape,train_numeric_movieId_features.shape,train_text_genres_features.shape,train_text_tag_features.shape,test_numeric_userId_features.shape,test_numeric_movieId_features.shape,test_text_genres_features.shape,test_text_tag_features.shape)
#torch.Size([82141, 1]) torch.Size([82141, 1]) torch.Size([82141, 25]) torch.Size([82141, 1720]) torch.Size([20536, 1]) torch.Size([20536, 1]) torch.Size([20536, 25]) torch.Size([20536, 1720])


# 定义模型
class ClassificationModel(nn.Module):
    def __init__(self, features,numeric_userId_features,numeric_movieId_features, text_genres_features,text_tag_features,hidden_size=500, output_size=1):
        super(ClassificationModel, self).__init__()

        self.fc1 = nn.Linear(features.shape[1], hidden_size)
        self.fc2 = nn.Linear(hidden_size * 4, hidden_size*2)
        self.fc5 = nn.Linear(hidden_size*2, hidden_size)
        self.fc6 = nn.Linear(hidden_size * 5, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

        self.fc3_numeric_userId_features = nn.Linear(numeric_userId_features.shape[1], hidden_size)
        self.fc3_numeric_movieId_features = nn.Linear(numeric_movieId_features.shape[1], hidden_size)
        self.fc3_text_genres_features = nn.Linear(text_genres_features.shape[1], hidden_size)
        self.fc3_text_tag_features = nn.Linear(text_tag_features.shape[1], hidden_size)

        #Batch Normalization（批归一化）可以应用于不同维度的数据。
        # 在PyTorch中，nn.BatchNorm1d用于对一维数据进行批归一化，
        # 而nn.BatchNorm2d用于对二维数据进行批归一化。
        self.b1 = nn.BatchNorm1d(hidden_size)
        self.b2 = nn.BatchNorm1d(hidden_size*2)
        self.b5 = nn.BatchNorm1d(hidden_size)

        self.dropout = nn.Dropout(p=0.08)

        self.relu = nn.ReLU()

        #调整参数的权重
        self.weight_numeric_userId_features = nn.Parameter(torch.tensor([0.2]))
        self.weight_numeric_movieId_features = nn.Parameter(torch.tensor([0.2]))
        self.weight_text_genres_features = nn.Parameter(torch.tensor([0.55]))
        self.weight_text_tag_features = nn.Parameter(torch.tensor([0.05]))

    def forward(self, x, numeric_userId_features,numeric_movieId_features, text_genres_features,text_tag_features):

        out_numeric_userId_features = self.fc3_numeric_userId_features(numeric_userId_features) * self.weight_numeric_userId_features
        out_numeric_movieId_features = self.fc3_numeric_movieId_features(numeric_movieId_features) * self.weight_numeric_movieId_features
        out_text_genres_features = self.fc3_text_genres_features(text_genres_features) * self.weight_text_genres_features
        out_tag_genres_features = self.fc3_text_tag_features(text_tag_features) * self.weight_text_tag_features

        out = torch.cat([out_numeric_userId_features, out_numeric_movieId_features,out_text_genres_features,out_tag_genres_features], dim=1)
        out = self.fc2(out)
        out = self.b2(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.fc5(out)
        out = self.b5(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.fc3(out)

        return out

model = ClassificationModel(features,numeric_userId_features,numeric_movieId_features, text_genres_features,text_tag_features)
model.to(device)
# 定义损失函数和优化器
criterion = nn.L1Loss().to(device)
#优化器加上学习率以及惩罚率  两者的参数可以调整寻找到最佳
optimizer = optim.Adam(model.parameters(), lr=0.01,weight_decay=0.002)
scheduler = StepLR(optimizer, step_size=10, gamma=0.92)

# 训练模型
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    # 前向传播
    outputs = model(train_features,train_numeric_userId_features,train_numeric_movieId_features,train_text_genres_features,train_text_tag_features)
    # 计算损失
    loss = criterion(outputs, train_target.float())
    # 反向传播
    loss.backward()
    # 更新参数
    optimizer.step()
    # 在每个 batch 后更新学习率
    scheduler.step()

    # 每10个epoch打印一次损失
    if (epoch+1) % 10 == 0:
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

# 在测试集上进行预测
model.eval()  # 设置模型为评估模式，即预测模式

with torch.no_grad():
    test_outputs = model(test_features,test_numeric_userId_features,test_numeric_movieId_features,test_text_genres_features,test_text_tag_features)
    #将计算的结果转换为dataframe
    df = pd.DataFrame(test_outputs.numpy())
    df2= pd.DataFrame(test_target.numpy())
    excel_file = 'archive/predict.csv'
    df.to_csv(excel_file)
    #excel_file = 'archive/true.csv'
    #df2.to_csv(excel_file)
    the_total = df.shape[0]
    set_interval = 0.25
    set_num = 0
    first_column = df.iloc[:, 0]
    second_column = df2.iloc[:, 0]
    print(f"max :{first_column.max()}",f"    realmax :{second_column.max()}")
    print(f"min : {first_column.min()}",f"    realmin :{second_column.min()}")
    #print(the_total)
    #print(first_column)
    #print(second_column)
    for i in range(the_total):
        if abs(first_column[i] - second_column[i]) <= set_interval:
            set_num = set_num + 1
    print('Test Accuracy: ',set_num / the_total)
