---
title: Deep Learning 学习笔记
date: 2024-05-08 20:34:37
mathjax: true
top: 90
categories: # 分类
- [Math]
- [Deep-Learning]
tags: # 标签
- 深度学习
---

# Pytorch神经网络基础

## 自动求导

<!--more-->

<img src="Data-flow.png" width="50%" height="50%" title="数据流" alt="错误无法显示"/>

在如图所示的数据流下，要计算$y$关于$x$的梯度，可以采样两种方法

### 正向传播（Forward Propagation）
梯度计算方向和数据流方向相同：$\frac{ dy } { dx } = \frac{ {dy} } { {db} } \left( {\frac{ {db} } { {da} }\left( {\frac{ {da} } { {dx} } } \right)} \right)$，称为正向传播模式

### 反向传播（Backward Propagation）
梯度计算方向和数据流方向相反：$\frac{ dy } { dx } = \frac{ {da} } { {dx} } \left( {\frac{ {db} } { {da} }\left( {\frac{ {dy} } { {db} } } \right)} \right)$，称为反向传播模式

Back Propagation算法是多层神经网络的训练中举足轻重的算法。简单的理解，它的确就是复合函数的链式法则，但其在实际运算中的意义比链式法则要大的多。

<img src="Backward-propagation-example.png" width="50%" height="50%" title="反向传播示例" alt="错误无法显示"/>

以上图为例子，我们相求$e$关于$a$和$b$的导数，那么我们有$\frac{ {de} }{ {da} } = \frac{ {de} }{ {dc} }\frac{ {dc} }{ {da} }$ 和 $\frac{ {de} }{ {db} } = \frac{ {de} }{ {dc} }\frac{ {dc} }{ {db} } + \frac{ {de} }{ {dd} }\frac{ {dd} }{ {db} }$

如果采用Forward Propagation，我们会发现这样做是十分冗余的，因为很多路径被重复访问了。比如图中的a-c-e和b-c-e就都走了路径c-e。对于权值动则数万的深度模型中的神经网络，这样的冗余所导致的计算量是相当大的。

Backward Propagation算法则机智地避开了这种冗余，它对于每一个路径只访问一次就能求顶点对所有下层节点的偏导值。正如反向传播(BP)算法的名字说的那样，BP算法是反向(自上往下)来寻找路径的，从最上层的节点e开始，初始值为1，以层为单位进行处理。对于e的下一层的所有子节点，将1乘以e到某个节点路径上的偏导值，并将结果“堆放”在该子节点中。等e所在的层按照这样传播完毕后，第二层的每一个节点都“堆放"些值，然后我们针对每个节点，把它里面所有“堆放”的值求和，就得到了顶点e对该节点的偏导。然后将这些第二层的节点各自作为起始顶点，初始值设为顶点e对它们的偏导值，以"层"为单位重复上述传播过程，即可求出顶点e对每一层节点的偏导数。

而神经网络正是需要对每一层求梯度，因此BP算法恰好契合了神经网络的需要。

## 模型构造
### 层和块

Pytoch中Module是一个很重要的概念，Module可以认为是任何一个层和一个神经网络它都属于Module的一个子类.在PyTorch中，nn.Module类的子类可以像函数一样被调用，这是因为在nn.Module的实现中，__call__方法被重写了，允许你像调用函数一样调用它们。当你调用一个继承自nn.Module的类的实例时，PyTorch会自动调用forward方法，这个方法定义了这个模型的前向传播逻辑。

``` python
import torch
from torch import nn
from torch.nn import functional as F

X = torch.rand(2, 20)

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.out = nn.Linear(256, 10)

    def forward(self, X):
        # nn.Relu()是构造了一个ReLU对象，并不是函数调用，而F.ReLU()是函数调用
        return self.out(F.relu(self.hidden(X)))

net1 = nn.Sequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))
net2 = MLP()
print(net1(X))
print(net2(X))
```

这个例子中我们通过自定义继承nn.Module这个类来实现了特定功能的函数，我们可以通过继承nn.Module这个类可以比Sequential去更灵活的去定义我们的参数是什么样子以及如何做前向计算。

比如一个混合搭配各种混合块的例子：

``` python
class NestMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(20, 64), nn.ReLU(),
                                 nn.Linear(64, 32), nn.ReLU())
        self.linear = nn.Linear(32, 16)

    def forward(self, X):
        return self.linear(self.net(X))

chimera = nn.Sequential(NestMLP(), nn.Linear(16, 20))
print(chimera(X))
```

因此通过这种方法我们可以进行更加灵活的定义。

## 参数管理

* 访问某一层的参数

``` python
import torch
from torch import nn

net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
X = torch.rand(size=(2, 4))

print(net[2].state_dict())
print(type(net[2].bias))
print(net[2].bias)
print(net[2].bias.data) # 访问对应的值
print(net.state_dict()['2.bias'].data)
print(net[2].weight.grad) # 这里还没有做反向计算，所以梯度为None
```

* 一次性访问所有参数

``` python
print(*[(name, param.shape) for name, param in net[0].named_parameters()]) # *[]序列解释包，将这个列表解包成单独的元组
print(*[(name, param.shape) for name, param in net.named_parameters()])
```

## 初始化

初始化的目的是让模型在一开始的时候使得每一层的输入输出大小在一个尺度上面，不要然它出现越往后面越大或者越往后面越小的情况，使模型爆掉了。只要初始化开始时不出问题，不同的初始化方法对精度的影响其实差不多。

### 一般初始化方法

遍历所有Module进行初始化

``` python
def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=0, std=0.01)
        nn.init.zeros_(m.bias)

net.apply(init_normal)
print(net[0].weight.data, net[0].bias.data)

def init_constant(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 1)
        nn.init.zeros_(m.bias)

net.apply(init_constant)
print(net[0].weight.data, net[0].bias.data)
```

### 对某些块应用不同的初始化方法

``` python
def xavier(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)

def init_42(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 42)

net[0].apply(xavier)
net[2].apply(init_42)
print(net[0].weight.data[0])
print(net[2].weight.data)
```

### 自定义初始化

也可以自定义初始化方法，如初始化保留绝对值大于等于5的权重

``` python
def my_init(m):
    if type(m) == nn.Linear:
        print(
            "Init",
            *[(name, param.shape) for name, param in m.named_parameters()][0]
        )
        nn.init.uniform_(m.weight, -10, 10)
        m.weight.data *= m.weight.data.abs() >= 5

net.apply(my_init)
print(net[0].weight[:2])
```

### 更暴力的方法

还可以直接把值拿出来做替换，如：

``` python
net[0].weight.data[:] += 1
net[0].weight.data[0, 0] += 42
net[0].weight.data[0]
```
### 参数绑定

当有一些layer想要sharing某些weight时，可以进行参数绑定，也就是在构建net时其指向同一个类，这是不用网络直接共享权重的一个方法

``` python
shared = nn.Linear(8, 8)
net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), shared, nn.ReLU(), shared, nn.ReLU(), nn.Linear(8, 1))
print(net[2].weight.data[0] == net[4].weight.data[0])
net[2].weight.data[0, 0] = 100
print(net[2].weight.data[0] == net[4].weight.data[0])
```

## 自定义层

本质上讲，自定义层和自定义网络没什么本质区别，因层也是nn.Module的一个子类

### 构造一个没有任何参数的自定义层

``` python
import torch
import torch.nn.functional as F
from torch import nn

class CenteredLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        return X - X.mean()

layer = CenteredLayer()
print(layer(torch.FloatTensor([1, 2, 3, 4, 5])))
```
进而可以将层作为组件合并到构建更复杂的模型中
``` python
net = nn.Sequential(nn.Linear(8, 128), CenteredLayer())
Y = net(torch.rand(4, 8))
print(Y)
```

### 自定义带参数的层

``` python
class MyLinear(nn.Module):
    def __init__(self, in_units, units):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_units, units))
        self.bias = nn.Parameter(torch.randn(units, ))
    
    def forward(self, X):
        linear = torch.matmul(X, self.weight.data) + self.bias.data
        return F.relu(linear)

dense = MyLinear(5, 3)
print(dense.weight)
print(torch.rand(2, 5))
```

### 使用自定义层构建模型

同样可以使用自定义层构建模型

``` python
net = nn.Sequential(MyLinear(64, 8), MyLinear(8, 1))
print(net(torch.rand(2, 64)))
```

## 读写文件

### 加载和保存张量

``` python
import torch
from torch import nn
from torch.nn import functional as F

# 储存一个张量列表，然后把它们读回内存
x = torch.arange(4)
y = torch.zeros(4)
torch.save([x, y], 'x-file')
x2, y2 = torch.load('x-file')
print((x2, y2))

# 写入或读取从字符串映射到张量的字典
mydict = {'x': x, 'y': y}
torch.save(mydict, 'mydict')
mydict2 = torch.load('mydict')
print(mydict2)
```

### 加载和保存模型参数

我们可以通过state_dict()来得到所有的Parameter中字符串到Parameter值的一个映射，并将其保存下来实例化一个模型的备份。

#### 模型参数保存

``` python
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.output = nn.Linear(256, 10)
    
    def forward(self, x):
        return self.output(F.relu(self.hidden(x)))

net = MLP()
X = torch.randn(size=(2, 20))
Y = net(X)
torch.save(net.state_dict(), 'mlp.params')
```

#### 模型参数读取

``` python
clone = MLP()
clone.load_state_dict(torch.load('mlp.params'))
print(clone.eval())
Y_clone = clone(X)
print(Y_clone == Y)
```

## 模型训练步骤

### 基本库导入

``` python
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from d2l import torch as d2l
from torch.utils.data import TensorDataset, Dataset, DataLoader
```

### Dataset数据集构建

在拿到一个张量数据后，首先要将其整理成Dataset的形式，首先要划分输入数据(features)和输出数据(labels)，然后将其整理为Dataset的形式，Dataset本身不提供数据的批处理或迭代功能。

#### 使用默认Dataset形式

``` python
dataset = TensorDataset(features, labels)
```

#### 使用自定义Dataset形式

除了使用内置的数据集，我们也可以自定义数据集。自定义数据集需要继承Dataset类，并实现__len__和__getitem__两个方法。在实际应用中，self.data 可以是任何类型的数据结构，只要能够按照索引获取样本即可。

``` python
class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        # 返回一个包含特征和标签的元组
        return self.features[idx], self.labels[idx]
```

#### Dataset数据查看

Dataset的访问方法为按照样本的索引访问单个样本，常用的操作为

* 访问第$i$个样本的features和labels\
``` python
dataset[i]
```

* 访问第$i$个样本的features或labels\
``` python
dataset[i][0]
dataset[i][1]
```

* 访问Dataset前$m$个样本的的features和labels\
``` python
m = 100
for i, data in enumerate(dataset):
    if i >= m:  # 如果已经打印了100个样本，跳出循环
        break
    print(f"Sample {i}: {data}")
```

### 数据预处理

自定义数据集类通常还需要进行数据预处理，例如归一化、编码、格式化等，以确保数据适合模型训练。可通过sklearn中的方法对数据进行标准化处理。

``` python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
# fit_transform按数据特征列进行标准化
data_normal = scaler.fit_transform(data)
# transform按数据特征列进行标准化
test_data_normal = scaler.transform(testdata)
# inverse_transform逆标准化还原数据
data_row = scaler.inverse_transform(data_normal)
```

### 划分训练集、测试集、验证集

``` python
train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = \
    torch.utils.data.random_split(dataset, [train_size, val_size, test_size])
```
### DataLoader构建

DataLoader提供了一种便捷的方式来以批次的形式访问数据，它在内部实现了数据的迭代，可以按批次返回数据，同时还可以进行数据打乱和多线程数据加载。

#### Dataset数据集转换为DataLoader

``` python
batch_size = 10
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
```

DataLoader 中，shuffle 参数指定了是否在每个 epoch（数据加载周期）开始时对数据集进行打乱。具体来说，当 shuffle=True 时，整个数据集的顺序会被随机打乱，然后再划分为批次。

#### DataLoader访问方式

* 迭代访问，DataLoader 本身是一个迭代器，可以直接在它上面进行迭代，以按批次获取数据。每次迭代返回的是一个数据批次，通常是一个包含特征和标签的元组。\
``` python
for batch_index, (features, labels) in enumerate(data_loader):
    # 在这里使用 features 和 labels 进行模型训练或评估
for features, labels in data_loader:
```

* 按索引访问\
``` python
single_batch = next(iter(data_loader)) 
```

* 使用 len() 函数，可以使用内置的 len() 函数来获取 DataLoader 中的批次总数。\
``` python
num_batches = len(data_loader)
```

* 结合 Subset 使用，当需要从一个完整的数据集中选择一个子集进行训练或验证时，可以使用 Subset 随机选择或指定一系列索引。\
``` python
from torch.utils.data import Subset
indices = [...]  # 指定的索引列表
subset = Subset(full_dataset, indices)
data_loader = DataLoader(subset, ...)
```

### 模型训练

在每个迭代周期里，我们将完整遍历一次数据集（train_data），不停地从中获取一个小批量的输入和相应的标签。对于每⼀个小批量，我们会进⾏以下步骤:

* 通过调用net(X)生成预测并计算损失Loss（前向传播）。
* 通过进行反向传播来计算梯度。
* 通过调用优化器来更新模型参数。

#### 基本参数定义
``` python
num_epochs, lr = 10,  1e-2
trainer = torch.optim.SGD(net.parameters(), lr=lr)
loss = nn.MSELoss()
# nn.MSELoss()默认关键字reduction="mean"，求均方误差，返回一个标量
# reduction="none"：求所有对应位置的差的平方，返回的仍然是一个和原来维度一样的tensor。
# reduction="sum"：求所有对应位置差的平方的和，返回的是一个标量。
```

#### 训练过程

``` python
for epoch in range(num_epochs):
    for X, y in train_loader:
        l = loss(net(X), y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    # 打印训练信息
    l = loss(net(features), labels)
    print(f'Epoch {epoch+1}/{num_epochs} Loss: {l.item()}')
```

## 模型调参

### 学习率

学习率决定了在每步参数更新中，模型参数有多大程度（或多快、多大步长）的调整。学习率是一个超参数。不同学习率的影响可以用下图表示

<img src="learning-rate-effect.png" width="50%" height="50%" title="学习率影响" alt="错误无法显示"/>


学习率还会跟优化过程的其他方面相互作用，这个相互作用可能是非线性的。小的batch size最好搭配小的学习率，因为batch size越小也可能有噪音，这时候就需要小心翼翼地调整参数。

# GPU部署

## GPU 可用性检查

* shell中查看GPU使用率
``` python
nvidia-smi
```

* 检查GPU是否可用
``` python
print(torch.cuda.is_available())
```

## 指定GPU设备
深度学习的所有框架都是默认在CPU上做运算的，使用GPU的话需要先指定GPU

### GPU上的张量运算

* 查看张量所在的设备
``` python
x = torch.tensor([1, 2, 3])
print(x.device)
```

* 数据在GPU上储存\
可以采用两种方法将数据储存在GPU上
``` python
x = x.cuda()
print(x.device)
X = torch.ones(2, 3, device='cuda:0')
print(X.device)
print(x+X)
```
值得注意的是在GPU上做运算必须要求数据在同一个设备(GPU)上

### 神经网络与GPU

同样可以将NN部署在GPU上

```python
net = nn.Sequential(nn.Linear(3, 1))
net = net.to(device='cuda:0')
print(net(X))
# 确认模型参数储存在同一个GPU上
print(net[0].weight.data.device)
```

# Standard Examples

## DL Model based on GPU

### 查看GPU设备信息并指定所用GPU

``` python
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import TensorDataset, Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from mypackage import mydl
from d2l import torch as d2l

def check_gpu_info():
    '显示 GPU 版本和内存信息'
    if not torch.cuda.is_available():
        print("No GPU found. CPU will be used.")
        return
    
    # 使用 nvidia-smi 命令获取 GPU 版本信息
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, check=True)
        print("GPU Version Info:")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print("Failed to run nvidia-smi command:", e)
    
    # 使用 PyTorch API 获取 GPU 内存信息
    print("\nGPU Memory Info:")
    print(f"GPU Total Numbers: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(i)
        total_memory_gb = properties.total_memory / (1024 ** 3)
        used_memory_gb = torch.cuda.memory_allocated(i) / (1024 ** 3)
        print(f"GPU {i}:")
        print(f"\tName: {properties.name}")
        print(f"\tTotal Memory: {total_memory_gb:.2f} GB")
        print(f"\tUsed Memory: {used_memory_gb:.2f} GB")
check_gpu_info()
device = torch.device(f'cuda:{0}')
```

### 数据加载

``` python
###############################################################################
#                            Data Preprocessing
###############################################################################
filename = rf'.\Dataset\Traindata(r-u_to_k).xlsx'
data = pd.read_excel(filename, sheet_name="Sheet1", header=None)
data = torch.tensor(data[1:].values.astype(np.float32))
data = data[:, :3]
```

### 数标准化

数据标准化是为了使得模型更快收敛，通常用sklearn中的方法进行数据标准化，sklearn处理的对象是numpy数组，因此在使用前要注意将数据类型转换为numpy

``` python
###############################################################################
#                            Data Standardization
###############################################################################
scaler = StandardScaler()
data = torch.tensor(scaler.fit_transform(data.numpy()), dtype=torch.float32)
```

### 构建Dataset和Dataloader

``` python
###############################################################################
#               Building corresponding Dataset and Dataloader
###############################################################################
# Building Dataset
dataset_st = TensorDataset(data_st[:, 0:2], data_st[:, [2]])
train_size = int(0.8 * len(dataset_st))
val_size = int(0.1 * len(dataset_st))
test_size = len(dataset_st) - train_size - val_size
train_dataset_st, val_dataset_st, test_dataset_st = torch.utils.data.random_split(dataset_st, [train_size, val_size, test_size])
# Building Dataloader
batch_size = 10
train_loader = DataLoader(train_dataset_st, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset_st, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset_st, batch_size=batch_size, shuffle=True)
```

### 构建模型并进行初始化

``` python
###############################################################################
#                      Building Model and Inilization
###############################################################################
net = nn.Sequential(
    nn.Linear(2, 128), nn.ReLU(),
    nn.Linear(128, 32), nn.ReLU(),
    nn.Linear(32, 1)
)

def xavier(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)

net.apply(xavier)
```

### 在GPU上进行模型训练

``` python
###############################################################################
#                            Model Training
###############################################################################
num_epochs, lr = 50, 1e-2
optimizer = torch.optim.SGD(net.parameters(), lr=lr)
loss = nn.MSELoss()
net.to(device=device)
net.train()
print('training on', device)

for epoch in range(num_epochs):
    for i, (X, y) in enumerate(train_loader):
        optimizer.zero_grad()
        X, y = X.to(device), y.to(device)
        l = loss(net(X), y)
        l.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}/{num_epochs} Loss: {l.item()}')
```

### 模型评估与预测

``` python
###############################################################################
#                            Model Evaluation
###############################################################################
net.eval()

with torch.no_grad():
    y_pred = net(test_dataset_st[:][0].cuda())


pred_test_data_st = torch.cat([test_dataset_st[:][0], y_pred.cpu()], dim=1)
test_data_st = torch.cat([test_dataset_st[:][0], test_dataset_st[:][1]], dim=1)
```

### 数据可视化

matplotlib函数基于numpy数组进行处理，当有tensor会将自动转换为numpy格式，但是他只能处理cpu的数据，因此模型预测的数据结果需要移动到cpu上。

``` python
###############################################################################
#                            Figure Plotting
###############################################################################
Number = np.arange(len(y_pred)) + 1

fig, ax = plt.subplots(1, 1, figsize=(7, 5))
ax.scatter(Number,test_data_st[:, 2], label="Ture data", c='red')
ax.scatter(Number, pred_test_data_st[:, 2], label="DL predictions", c='blue')
ax.set_xlabel("u235(Standarization)")
ax.set_ylabel("keff(Standarization)")
ax.legend()

fig, ax = plt.subplots(1, 1, figsize=(7, 5))
ax.scatter(Number, scaler.inverse_transform(test_data_st.numpy())[:, 2], label="Ture data", c='red')
ax.scatter(Number, scaler.inverse_transform(pred_test_data_st.numpy())[:, 2], label="DL predictions", c='blue')
ax.set_xlabel("u235")
ax.set_ylabel("keff")
ax.legend()

plt.show()
```

## High Dimensional Linear Regression

### 数据产生

``` python
import numpy as np
import pandas as pd

# 设置随机种子以获得可重复的结果
np.random.seed(42)
num_inputs = 3
num_samples = 1000

# 生成输入数据，这里我们假设每个输入变量的范围是0到10
X1 = np.random.uniform(low=0, high=4, size=num_samples)
X2 = np.random.uniform(low=0, high=4, size=num_samples)
X3 = np.random.uniform(low=0, high=4, size=num_samples)

# 生成噪声项，这里我们假设噪声项是正态分布的
epsilon = np.random.normal(loc=0, scale=1, size=num_samples)

# 计算输出变量Y，包含非线性项
Y = X1**2 + 5*np.sin(X2) - X3**3 + epsilon

# 创建一个DataFrame
data = {
    'X1': X1,
    'X2': X2,
    'X3': X3,
    'Y': Y
}
df = pd.DataFrame(data)
df.to_csv('./Dataset/Traindata(r1-r3-u_to_k).csv', index=False)
```

### 回归模型训练
``` python
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import TensorDataset, Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from mypackage import mydl


###############################################################################
#                      Building Model and Inilization
###############################################################################
net = nn.Sequential(
    nn.Linear(3, 128), nn.ReLU(),
    nn.Linear(128, 32), nn.ReLU(),
    nn.Linear(32, 1)
)
mydl.init_cnn(net)


###############################################################################
#                            Data Preprocessing
###############################################################################
filename = rf'./Dataset/Traindata(r1-r3-u_to_k).csv'
data = mydl.read_data(filename, use_header=0)
data= torch.tensor(data.to_numpy(dtype=np.float32))


###############################################################################
#                            Data Standardization
###############################################################################
normalizer = mydl.Normalizer()
normalizer.fit(data)
data_st = normalizer.transform(data)


###############################################################################
#               Building corresponding Dataset and Dataloader
###############################################################################
# Dataset Building
dataset_st = TensorDataset(data_st[:, 0:3], data_st[:, [3]])
train_size = int(0.8 * len(dataset_st))
val_size = int(0.1 * len(dataset_st))
test_size = len(dataset_st) - train_size - val_size
train_dataset_st, val_dataset_st, test_dataset_st = \
    torch.utils.data.random_split(dataset_st, [train_size, val_size, test_size])
# Dataloader Building
batch_size = 200
train_loader = DataLoader(train_dataset_st, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset_st, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset_st, batch_size=batch_size, shuffle=True)


###############################################################################
#                            Model Training
###############################################################################
device = mydl.try_gpu()
num_epochs, lr = 300, 1e-3
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
loss = nn.MSELoss()
net.to(device=device)
print('training on', device)

Epoch, train_losses, val_losses= [], [], []
for epoch in range(num_epochs):
    net.train()
    for i, (X, y) in enumerate(train_loader):
        optimizer.zero_grad()
        X, y = X.to(device), y.to(device)
        l = loss(net(X), y)
        l.backward()
        optimizer.step()

    if (epoch + 1) % 5 == 0:
        train_loss = mydl.evaluate_loss(net, train_loader, loss)
        val_loss = mydl.evaluate_loss(net, test_loader, loss)
        Epoch.append(epoch + 1)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(f'Epoch {epoch+1}/{num_epochs} Loss: {l.item()}')

with torch.no_grad():
    test_inputs = test_dataset_st[:][0].to(device)
    test_targets = test_dataset_st[:][1].to(device)
    test_outputs = net(test_inputs)


###############################################################################
#                            Data Anti-normalization
############################################################################### 
test_targets = test_targets * normalizer.std[3] + normalizer.mean[3]
test_outputs = test_outputs * normalizer.std[3] + normalizer.mean[3]
print(normalizer.std, normalizer.mean)
###############################################################################
#                               Figure
###############################################################################
# Loss in every Eopch
fig, ax = plt.subplots(1, 1, figsize=(6, 5))
ax.plot(Epoch, train_losses, label='Train Loss')
ax.plot(Epoch, val_losses, label='Validation Loss')
ax.set_xlabel('Epoch')
ax.set_ylabel('MSE Loss')
ax.set_title('Loss Curves')
ax.set_yscale('log')
ax.legend(framealpha=0)

# Prediction Accuracy
fig, ax = plt.subplots(1, 1, figsize=(6, 5))
ax.scatter(test_targets.to(mydl.cpu()), test_outputs.to(mydl.cpu()), s=5, c='red')
m = torch.max(torch.abs(test_targets)).to(mydl.cpu())
ax.plot([-m, m], [-m, m])
ax.set_xlim([-m, m])
ax.set_ylim([-m, m])
ax.set_xlabel('Target Values')
ax.set_ylabel('Prediction Values')
plt.show()
```

# 数据集类型

## DataFrame数据集

pandas 的 read_excel 函数用于读取Excel文件，并将数据加载到一个 DataFrame 对象中。DataFrame 对象本身不会在数据中显示行号和列号，但它们是 DataFrame 的一部分，打印时会看到对应的行号和列号。


# CNN (Convolutional Neural Network)

适合于计算机视觉的神经⽹络架构基于两个原则：

1. 平移不变性（translation invariance）：不管检测对象出现在图像中的哪个位置，神经网络的前面几层
应该对相同的图像区域具有相似的反应。

2. 局部性（locality）：神经网络的前面几层应该只探索输⼊图像中的局部区域，而不过度在意图像中相隔较远区域的关系，这就是“局部性”原则。最终，可以聚合这些局部特征，以在整个图像级别进行预测。

## 什么是卷积？

### 从蝴蝶效应说起

{% gp 2-2 %}
    <img src="convolution-picture1.png" width="40%" title="卷积说明1" alt="错误无法显示"  />
    <img src="convolution-picture2.png" width="40%" title="卷积说明2" alt="错误无法显示" />
{% endgp %}

为了更好的理解卷积，考虑这样一个例子，有一只蝴蝶不停地在扇动翅膀，不同时刻扇动翅膀的快慢不同，因此其产生的破坏效果也不同，其破坏效果的影响力我们用左图来描述，并且某时刻产生的破坏效果不会马上消失，而是随时间逐渐衰减，其衰减效果如右图所示。

现在解决一个问题，求一下$t$时刻感受到的破坏力？这个问题也很简单，把之前所有时刻对$t$时刻的影响加起来就行了，本质上也就是求这个式子:

$$
\int_0^t {f\left( x \right)g\left( {t - x} \right)dx}
$$

这也就是我们后面要说的卷积。

### 卷积、卷积 为什么叫“卷积”？

我们给出所谓的卷积的定义，也就是这个式子
$$
\int_{ - \infty }^\infty {f\left( \tau \right)g\left( {x - \tau} \right)d\tau}
$$

从左边图可以看出，图中每一条连线都对应着一对$f(x)$和$g(t-x)$的相乘，把所有的值加起来，就得到了我们所谓的卷积。

{% gp 2-2 %}
    <img src="convolution-picture3.png" width="40%" title="卷积说明3" alt="错误无法显示"  />
    <img src="convolution-picture4.png" width="40%" title="卷积说明4" alt="错误无法显示" />
{% endgp %}

此时如果我们将$g(t)$函数翻转一下，会发现卷积实际上就是将函数翻转后对应位置相乘求和，这也就是为什么叫卷积。

### 什么是图像的卷积操作

如果我们把视野放得更广一点，在上面蝴蝶效应的例子中，如果影响力的变化不是随时间改变，而是随着空间距离而改变的，也就是说对$x$位置产生影响的是其他很多位置，那么回到开始的问题，什么是图像的卷积操作？图像的卷积操作实际上就是去看图像上其他很多像素点对一个像素点是如何产生影响的，举个例子

<img src="convolution-picture5.png" width="90%" title="平滑卷积核操作" alt="错误无法显示" />

可以看到这个例子中，卷积核规定了周围的像素点对当前像素点的影响，当前在经过一个与平滑卷积核进行卷积操作后，对图像进行了平滑，也就是说在这个卷积核下考虑周围像素点对某个像素点影响，遍历整个图片后，得到的结果是每个像素点更平滑。

{% gp 3-3 %}
    <img src="convolution-picture6.png" width="40%" title="卷积说明6" alt="错误无法显示" />
    <img src="convolution-picture7.png" width="40%" title="卷积说明7" alt="错误无法显示" />
    <img src="convolution-picture8.png" width="40%" title="卷积说明8" alt="错误无法显示" />
{% endgp %}

我们现在考虑$g(m,n)$这个卷积核下，$(x,y)$周围的像素点对$(x,y)$这个像素点的影响效果，根据卷积的定义，可以得到

$$
\begin{array}{l}
\begin{aligned}
f\left( {x,y} \right)g\left( {m,n} \right) &= \sum {f\left( {x,y} \right)g\left( {m - x,n - y} \right)} \\
& = f\left( {x - 1,y - 1} \right)g\left( {1,1} \right) + f\left( {x,y - 1} \right)g\left( {0,1} \right) + f\left( {x + 1,y - 1} \right)g\left( { - 1,1} \right)\\
& + f\left( {x - 1,y} \right)g\left( {1,0} \right) + f\left( {x,y} \right)g\left( {0,0} \right) + f\left( {x + 1,y} \right)g\left( { - 1,0} \right)\\
& + f\left( {x - 1,y + 1} \right)g\left( {1, - 1} \right) + f\left( {x,y + 1} \right)g\left( {0, - 1} \right) + f\left( {x + 1,y + 1} \right)g\left( { - 1, - 1} \right)
\end{aligned}
\end{array}
$$

<img src="convolution-picture9.png" width="90%" title="卷积说明8" alt="错误无法显示" />

同样我们发现它仍然是卷着乘的，我们我们将它翻转$180^\circ$后会发现是对应位置相乘，实际上后来我们CNN中用的卷积核就是翻转后的结果，它可以直接扣在图像上直接相乘再相加，但它本质上仍然是一个卷积运算。

### 卷积神经网络与卷积

卷积神经网络主要是用来干图像识别的。

### 参考链接

<iframe src="//player.bilibili.com/player.html?aid=418492547&bvid=BV1VV411478E&cid=353587154&p=1&autoplay=0" scrolling="no" border="0" frameborder="no" framespacing="0" allowfullscreen="true" > </iframe>

## 再谈全连接层与卷积

在之前处理图片时，我们将一张图片reshape成一个1D向量来处理，现在我们考虑它的空间信息，对于一个图片它都包含一部分空间信息，因此我们选择用矩阵(宽度, 高度)去描述神经网络的输入$x$和输出$h$，对应地我们可以将我们的权重变为4D张量，此时有对应的变换关系

$$
{ h_{ij} } = \sum\limits_{k,l} { {w_{ijkl} } {x_{kl} } } 
$$

接下来我们对$w$做一个重新的索引，使得${v_{ijab} } = {w_{ij(i + a)(j + b)} }$，此时有

$$
\begin{equation}
{ h_{ij} } = \sum\limits_{k,l} { {w_{ijkl} } {x_{kl} } }  = \sum\limits_{a,b} { {v_{ijab} } {x_{(i + a)(j + b)} } }
\label{convolution}
\end{equation}
$$

这个式子可以看成$(i, j)$位置的输出$h$是由$(i, j)$位置的周边$(i+a, j+b)$的一些输入$x$在权重${\bf v}$下所共同影响而得到的。下面我们根据我们的基本原则引出我们的卷积：

### 平移不变性

在方程\eqref{convolution}中，权重${\bf v}$本质上就是一组识别器，而这时如果$(i, j)$发生变化(对应平移)，这时权重$v_{ijab}$也会发生变化，使得输出结果做出对应的改变，而根据平移不变性的要求，不管检测对象出现在图像中的哪个位置，神经网络的前面几层应该对相同的图像区域具有相似的反应。所以我们并不希望发生平移时（也就是改变$i, j$时）输出也随$i, j$的变化而改变，因此我们说${\bf v}$并不该依赖于$(i, j)$，进而我们有$v_{ijab}=v_{ab}$，故而得到：

$$
{h_{ij} } = \sum\limits_{a,b} { {v_{ab} } {x_{(i + a)(j + b)} } }
$$

这就是所谓的二维交叉相关。

### 全连接层 VS 卷积

* 全连接层最大的问题在于全连接层第一层的权重参数矩阵的行取决于输入的维度，特别是在处理图像问题的时候，如果把每个像素点作为一个维度，这个模型会非常大，容易爆掉。

* 相比于全连接层，卷积的优势在于不管它的输入是多大，卷积核的大小总是固定的，这样就极大地降低了模型复杂度。

### 局部性

局部性的意思是说当我们评估$h_{ij}$时，我们不应该使用远离$x_{ij}$的参数，因此我们选择当$|a|, |b| > \Delta$时，使$v_{ab}=0$，此时有：
$$
{h_{ij} } = \sum\limits_{a =  - \Delta }^\Delta  {\sum\limits_{b =  - \Delta }^\Delta  { {v_{ab} } {x_{(i + a)(j + b)} } } }
$$

因此我们对全连接层使用平移不变性和局部性就得到了我们的卷积层，换句话说就是卷积是一个特殊的全连接层。

## 卷积层

卷积层本质是将输入和核矩阵进行交叉相关，加上偏移后得到输出，核矩阵和偏移都是可学习的参数，核矩阵的大小是超参数。

### 二维卷积层

<img src="2D_convolution.png" width="50%" height="50%" title="二维卷积层示例" alt="错误无法显示"/>

在二维互相关运算中，卷积窗口从输入张量的左上角开始，从左到右、从上到下滑动。当卷积窗口滑动到新一个位置时，包含在该窗口中的部分张量与卷积核张量进行按元素相乘，得到的张量再求和得到⼀个单⼀的标量值，由此我们得出了这⼀位置的输出张量值。

卷积核的宽度和高度大于1，而卷积核只与图像中每个大小完全适合的位置进行互相关运算，这一过程的数学表述可以表示为

$$
{\bf Y} = {\bf X} \star {\bf W} + b
$$

* 输入${\bf X}$: $n_{h} \times n_{w}$
* 卷积核${\bf W}$: $k_{h} \times k_{w}$
* 偏差: $b \in \mathbb{R}$
* 输出${\bf Y}$: $(n_{h} - k_{h} + 1) \times (n_{w} - k_{w} + 1)$

其中$\star$表示交叉相关运算，${\bf W}$和$b$是可学习的参数。

### 交叉相关 VS 卷积

* 二维交叉相关

$$
{h_{ij} } = \sum\limits_{a,b} { {v_{ab} } {x_{(i + a)(j + b)} } }
$$

* 二维卷积

$$
{h_{ij} } = \sum\limits_{a,b} { {v_{-a, -b} } {x_{(i + a)(j + b)} } }
$$

它们唯一的区别是卷积在索引$w$的时候是翻转的，相当于先翻转$180^\circ$再做交叉相关操作，这也是为什么称之为“卷积”。由于对称性的存在，在实际使用过程中它们没有任何区别，用二维交叉学出来的东西翻转过来就是用二维卷积学出来的东西。

### 卷积层的填充padding与步幅stride

填充和步幅是卷积层的超参数，填充是在输入周围添加额外的行和列，来控制输出形状的减少量，步幅是每次滑动核窗口时的步长，可以成倍的减少输出形状。

#### 填充

经过一层卷积操作后，$n_{h} \times n_{w}$的输入减小为为$(n_{h} - k_{h} + 1) \times (n_{w} - k_{w} + 1)$的输出，如果想做比较深的神经网络，我们就需要对其进行填充

<img src="convolution-fill.png" width="40%" height="50%" title="填充操作" alt="错误无法显示"/>

通过在输入的四周添加额外的行和和列，可以使得输出形状保持不变

* 填充$p_h$行和$p_w$列，输出形状为$(n_{h} - k_{h} + p_{h} + 1) \times (n_{w} - k_{w} +p_{w} + 1)$
* 为了保持形状不变，通常取$p_{h} = k_{h} - 1$, $p_{w} = k_{w} - 1$
    * 当$k_{h}$为奇数时：在上下两侧填充$p_{h} / 2$
    * 当$k_{h}$为偶数时：在上侧填充$\lceil p_{h} / 2 \rceil$, 在下侧填充$\lfloor p_{h} / 2 \rfloor$

#### 步幅

当输入一个比较大的图片时，可以通过调整步幅的大小来减小输出

* 给定高度$s_h$和宽度$s_w$的步幅，输出形状为$\lfloor (n_{h} - k_{h} + p_{h}) / s_{h} + 1 \rfloor \times \lfloor (n_{w} - k_{w} +p_{w}) / s_{h} + 1 \rfloor$
* 如果$p_{h} = k_{h} - 1$, $p_{w} = k_{w} - 1$，则输出形状为$\lfloor (n_{h} - 1) / s_{h} + 1 \rfloor \times \lfloor (n_{w} - 1) / s_{h} + 1 \rfloor$
* 如果输入高度$n_{h}$和宽度$n_{w}$可以被步幅整除，则输出形状为$(n_{h}/s_{h}) \times (n_{w}/s_{w})$

### 多输入和多输出通道

对于彩色图像来讲可能有RGB三个通道，如果直接将其转换为灰度则会丢失信息。

#### 多输入通道

* 输入${\bf X}$: ${c_i} \times {n_h} \times {n_w}$
* 核${\bf W}$: ${c_i} \times {k_h} \times {k_w}$
* 输出${\bf Y}$: ${m_h} \times {m_w}$

$$
{\bf Y} = \sum\limits_{i = 0}^{ {c_i} } { { {\bf X}_{i,:,:} } \star { {\bf W}_{i,:,:} } }
$$

多输入通道每个通道都有一个卷积核，结果是所有通道卷积结果的和。

#### 多输出通道

* 输入${\bf X}$: ${c_i} \times {n_h} \times {n_w}$
* 核${\bf W}$: ${c_o} \times {c_i} \times {k_h} \times {k_w}$
* 输出${\bf Y}$: ${c_o} \times {m_h} \times {m_w}$

$$
{\bf Y}_{i,:,:}= {\bf X} \star { {\bf W}_{i,:,:,:} } \quad {\rm for} \quad i=1,\ldots,c_o
$$

无论有多少输入通道，到目前为止我们只用到了单输出通道，但实际上我们可以有多个三维卷积核，每个核都可以生成一个输出通道。

#### 为什么使用多输入和多输出

* 对于每一个输出通道，它都有一个卷积核去识别特定的模式，

<img src="cat.png" width="80%" height="50%" title="猫图像识别" alt="错误无法显示"/>

* 输入通道核识别并组合输入中的模式\
当把输出通道的结果传给下一次层的输入时，下一通道会进一步进行特征提取并进行组合，得到一个组合的模式识别。

#### $1 \times 1$ 卷积层

$k_h = k_w = 1$这个卷积层是一个特殊的卷积层，它不识别空间模式，并不会提取空间信息，而只是用来融合通道。本质上它相当于输入形状为$n_h n_w \times c_i$的${\bf X}$，权重为$c_i \times c_o$的${\bf K}$的全连接层。

#### 多输入和多输出通道总结

* 输入${\bf X}$: ${c_i} \times {n_h} \times {n_w}$
* 核${\bf W}$: ${c_i} \times {k_h} \times {k_w}$
* 偏差${\bf B}$: $c_o \times c_i$
* 输出${\bf Y}$: ${m_h} \times {m_w}$
* 计算复杂度: $O (c_i c_o k_h k_w m_h m_w)$

## 池化层 Pooling

通常当我们处理图像时，我们希望逐渐降低隐藏表示的空间分辨率、聚集信息，这样随着我们在神经网络中
层叠的上升，每个神经元对其敏感的感受野（输入）就越大。

而的机器学习任务通常会跟全局图像的问题有关（例如：图像是否包含⼀只猫呢？），所以我们最后⼀
层的神经元应该对整个输入的全局敏感。通过逐渐聚合信息，生成越来越粗糙的映射，最终实现学习全局表
示的目标，同时将卷积图层的所有优势保留在中间层。

池化层最终返回窗口中最大或平均值，它同样有窗口大小、填充和步幅作为超参数，能够缓解卷积层对位置的敏感性。

在Pytorch中默认步幅大小与池化窗口相同。

### 填充、步幅和多通道

* 池化层和卷积层类似，都具体填充和步幅
* 没有可学习的参数，不需要学kernel
* 在每个输入通道应用池化层以获得相应的输出通道，它只是处理一下数据
* 输出通道数 = 输入通道数

### 最大池化层

输出每个信号中最强的模式信号

### 平均池化层

将最大池化层中的“最大”操作替换为“平均”

## LeNet 经典卷积神经网络

<img src="LeNet.png" width="80%" height="50%" title="LeNet神经网络" alt="错误无法显示"/>

LeNet是早期成功的神经网络，先使用卷积层来学习图片空间信息，然后使用全连接层转到类别空间。

### LeNet Pytorch实现

``` python
import torch
from torch import nn
from d2l import torch as d2l

class Reshape(torch.nn.Module):
    def forward(self, x):
        return x.view(-1, 1, 28, 28)
    
net = torch.nn.Sequential(
    Reshape(), nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.ReLU(), 
    nn.AvgPool2d(kernel_size=2, stride=2), 
    nn.Conv2d(6, 16, kernel_size=5), nn.ReLU(),
    nn.AvgPool2d(kernel_size=2, stride=2), nn.Flatten(), 
    nn.Linear(16 * 5 * 5, 120), nn.ReLU(), 
    nn.Linear(120, 84), nn.ReLU(), 
    nn.Linear(84, 10)
)

X = torch.rand(size=(1, 1, 28, 28), dtype=torch.float32)
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__, 'output shape: \t', X.shape)

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)

def evaluate_accuracy_gpu(net, data_iter, device=None):
    if isinstance(net, torch.nn.Module):
        net.eval()
        if not device:
            device = next(iter(net.parameters())).device
    metric = d2l.Accumulator(2)
    for X, y in data_iter:
        if isinstance(X, list):
            X = [x.to(device) for x in X]
        else:
            X = X.to(device)
        y = y.to(device)
        metric.add(d2l.accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]

lr, num_epochs = 0.1, 10
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
```
## VGG块

VGG块的想法是n个卷积层和1一个池化层把封装成块。

* VGG使用可重复使用的卷积块来构建深度卷积神经网络
* 不同卷积块的个数和超参数可以得到不同复杂程度的变种

## NiN网络

无论是 LeNet 还是 AlexNet 网络，在卷积层输出的最后，其最后都通过一个Flatten层来展平卷积层的输出，然后再加两个全连接层进行分类预测，但实际是这个全连接层是十分占内存的。我们知道一个 $1 \times 1$ 的卷积层可以等效为一个全连接层

<img src="1-1-convolution.png" width="80%" title="1乘1卷积" alt="错误无法显示"  />
<img src="NiN-block.png" width="30%" title="NiN架构" alt="错误无法显示" />

通过用 $1 \times 1$ 的卷积层代替全连接层，减少了模型大小，最终得到NiN的架构为

<img src="NiN-Networks.png" width="80%" title="NiN Networks" alt="错误无法显示"  />

* 无全连接层
* 交替使用NiN块和步幅为2的最大池化层逐步减小高宽
    * 增大通道数
* 最后使用全局平均池化层得到输出
    * 其输入通道数是类别数
* NiN块使用卷积层加两个1x1卷积层
    * 后者对每个像素增加了非线性性
* NiN使用全局平均池化层来替代VGG和AlexNet中的全连接层
    * 不容易过拟合，更少的参数个数