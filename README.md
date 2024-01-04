# AI整体框架
# pyTorch
## 一、神经网络

### 神经元模型
#### 结构
- 神经元模型是一个包含输入，输出与计算功能的模型。输入可以类比为神经元的树突，而输出可以类比为神经元的轴突，计算则可以类比为细胞核。中间的箭头线。这些线称为“连接”。每个上有一个“权值”。
- 连接是神经元中最重要的东西。每一个连接上都有一个权重。一个神经网络的训练算法就是让权重的值调整到最佳，以使得整个网络的预测效果最好。
- 我们使用a来表示输入，用w来表示权值。一个表示连接的有向箭头可以这样理解：在初端，传递的信号大小仍然是a，端中间有加权参数w，经过这个加权后的信号会变成a*w，因此在连接的末端，信号的大小就变成了a*w。
- 在其他绘图模型里，有向箭头可能表示的是值的不变传递。而在神经元模型里，每个有向箭头表示的是值的加权传递。
- 神经元 ——>单元 / 节点
#### 效果
神经元模型的使用可以这样理解：
我们有一个数据，称之为样本。样本有四个属性，其中三个属性已知，一个属性未知。我们需要做的就是通过三个已知属性预测未知属性。
具体办法就是使用神经元的公式进行计算。三个已知属性的值是a1，a2，a3，未知属性的值是z。z可以通过公式计算出来。
这里，已知的属性称之为特征，未知的属性称之为目标。假设特征与目标之间确实是线性关系，并且我们已经得到表示这个关系的权值w1，w2，w3。那么，我们就可以通过神经元模型预测新样本的目标。

### 单层神经网络（感知器
1. 两个层次 ：输入 / 输出 
2. 需要计算的层次 ：计算层 ——>一个计算层
3. 产生效果：线性分类
4. 偏置节点：没有输入 ——常数


### 两层神经网络（多层感知器
#### 结构
- 增加中间层 两层计算层
- 隐藏层的参数矩阵的作用就是使得数据的原始坐标空间从线性不可分，转换成了线性可分。
- g(W(1) * a(1) + b(1)) = a(2);   g(W(2) * a(2) + b(2)) = z;
- 非线性分类任务
#### 训练
- 损失：机器学习模型训练的目的，就是使得参数尽可能的与真实的模型逼近。具体做法是这样的。首先给所有参数赋上随机值。我们使用这些随机生成的参数值，来预测训练数据中的样本。样本的预测目标为yp，真实目标为y。那么，定义一个值loss，这个值称之为损失（loss），我们的目标就是使对所有训练数据的损失和尽可能的小。
- 如果将先前的神经网络预测的矩阵公式带入到yp中（因为有z=yp），那么我们可以把损失写为关于参数（parameter）的函数，这个函数称之为损失函数（loss function）。下面的问题就是求：如何优化参数，能够让损失函数的值最小。
- 此时这个问题就被转化为一个优化问题。一个常用方法就是高等数学中的求导，但是这里的问题由于参数不止一个，求导后计算导数等于0的运算量很大，所以一般来说解决这个优化问题使用的是梯度下降算法。梯度下降算法每次计算参数在当前的梯度，然后让参数向着梯度的反方向前进一段距离，不断重复，直到梯度接近零时截止。一般这个时候，所有的参数恰好达到使损失函数达到一个最低值的状态。
- 在神经网络模型中，由于结构复杂，每次计算梯度的代价很大。因此还需要使用反向传播算法。反向传播算法是利用了神经网络的结构进行的计算。不一次计算所有参数的梯度，而是从后往前。首先计算输出层的梯度，然后是第二个参数矩阵的梯度，接着是中间层的梯度，再然后是第一个参数矩阵的梯度，最后是输入层的梯度。计算结束以后，所要的两个参数矩阵的梯度就都有了。
- 优化问题只是训练中的一个部分。机器学习问题之所以称为学习问题，而不是优化问题，就是因为它不仅要求数据在训练集上求得一个较小的误差，在测试集上也要表现好。因为模型最终是要部署到没有见过训练数据的真实场景。提升模型在测试集上的预测效果的主题叫做泛化（generalization），相关方法被称作正则化（regularization）。神经网络中常用的泛化技术有权重衰减等。

### 多层神经网络（深度学习


***

## 二、pytorch的整体框架

### 1.torch

#### (1).Tensor概念
张量 ，最基础的运算单位 ，一个多维矩阵，一个可以运行在gpu上的多维数据

#### (2).Tensor的创建
1. torch.FloatTensor(2,3)  / torch.FloatTensor([2,3,4,5])
2. torch.randn(2,3)    //2*3随机数
3. torch.range(1,10,2)  ——> tensor([1,3,5,7,9])
4. torch.zeros/ones/empty ——>全为0/全为1/定义大小的空tensor
5. 从NumPy数组创建
~~~python
import numpy as np
numpy_array = np.array([1, 2, 3])
tensor_from_numpy = torch.from_numpy(numpy_array)
~~~


#### (3).Tensor的运算
1. torch.abs/add/div/pow ——>绝对值/求和/求商/求幂
2. torch.clamp(变量，上边界，下边界)
3. torch.mm/mv ——>矩阵相乘/与矩阵*向量    //一维向量默认为列向量
4. .T 转置矩阵

***

### 2.torch数据读取
#### (1).dataset类
自定义数据集
~~~
import torch
from torch.utils.data import Dataset

# 自定义一个名称为MyDataset的数据集（继承pytorch内置的Dataset类）
class MyDataset(Dataset):
    # 重写构造函数（输入2个tensor类型的参数：数据/数据集合，数据对应的标签/集合）
    def __init__(self, data_tensor, target_tensor):
        self.data_tensor = data_tensor        
        self.target_tensor = target_tensor
        
    # 重写len方法：返回数据集大小
    def __len__(self):
        return self.data_tensor.size(0)  # 一般情况输进来的数据都是集合
    
    # 上重写getitem方法：基于索引，返回对应的数据及其标签，组合成1个元组返回

    def __getitem__(self, index):
        return self.data_tensor[index], self.target_tensor[index]
~~~
调用数据集
~~~
# 生成数据集和标签集 （数据元素长度=标签元素长度）
data_tensor = torch.randn(10, 3)        # 10行3列数据，可以理解为10个元素，每个元素个一维的3个元素列表
target_tensor = torch.randint(2, (10,)) # 标签是0或1的10个元素   # 对应语法：torch.randint(low, high, size）

# 将数据封装成自定义数据集的Dataset
my_dataset = MyDataset(data_tensor, target_tensor)

# 调用方法：查看数据集大小
print('Dataset size:', len(my_dataset))


# 使用索引调用数据
print('tensor_data[0]: ', my_dataset[0])
~~~

***

#### (2).dataloader类
~~~
from torch.utils.data import DataLoader
tensor_dataloader = DataLoader(dataset=xxx,     # 传入的数据集, 必须参数，必须是Dataset 类型
                               batch_size=x,    # int 类型，每个 batch 有多少个样本
                               shuffle=x,       # bool 类型，在每个 epoch 开始的时候，是否对数据进行重新打乱；
                               num_workers=x)   # int 类型，加载数据的进程数，0 意味着所有的数据都会被加载进主进程，默认为 0
~~~

***

### 3..torchvision
#### 数据调用
Torchvision 库就是常用数据集 + 常见网络模型 + 常用图像处理方法
torchvision.datasets所支持的所有数据集，它都内置了相应的数据集接口
~~~python
# 以MNIST为例
import torchvision
mnist_dataset = torchvision.datasets.MNIST(root='./data', //指定保存数据集位置
                                       train=True, //是否加载训练集数据
                                       transform=None, //图像预处理
                                       target_transform=None,//图像标签预处理
                                       download=True//True 自动下载)
~~~

***

#### torchvision.transforms
_图像处理工具_
##### 数据类型变换
- .ToTensor()  ——>将 PIL.Image 或 Numpy.ndarray 数据转化为Tensor格式
- .ToPILImage(mode=())  ——>将 Tensor 或 Numpy.ndarray 格式的数据转化为 PIL.Image  ；mode=none  ：输入数据维度1/2/3/4 ——>数据类型/LA/RGB/RGBA
##### 图像变换操作
1. 对PIL.Image和 对Tensor都支持的变换操作
- Resize
~~~python
torchvision.transforms.Resize(size, interpolation=2)(PIL Image对象/tensor对象) //size 若是 int整数 较短边匹配到size另一边按比例缩放
~~~
- .CenterCrop / .RandomCrop /.FiveCrop  中心剪裁 / 随机剪裁 / 从中心和四角剪裁
~~~
torchvision.transforms.CenterCrop(size)(PIL Image对象/tensor对象)
torchvision.transforms.RandomCrop(size, padding=None)(PIL Image对象/tensor对象)
torchvision.transforms.FiveCrop(size)(PIL Image对象/tensor对象)
~~~
- RandomHorizontalFlip  /  RandomVerticalFlip  以某概率随机水平翻转图像和某概率随机垂直翻转图像
~~~
torchvision.transforms.RandomHorizontalFlip(p=0.5)(PIL Image对象/tensor对象)
~~~
2. 仅针对Tensor的操作
- 标准化
1. 标准化是指每一个数据点减去所在通道的平均值，再除以所在通道的标准差
2. 三维数据的前2维，可以说是长宽（面积大小），第三维习惯称之为通道
3. 图像数据的通道指的是图像在不同颜色通道上的信息。在RGB图像中，通常有三个通道，分别代表红色（R）、绿色（G）和蓝色（B）。每个通道包含了图像在对应颜色上的信息，可以表示图像中不同颜色的亮度和色彩。在灰度图像中只有一个通道，代表了图像的亮度信息。

```
transforms.normalize(mean_vals, std_vals,inplace=False)
output = (input - mean) / std ；
mean:各通道的均值；
std：各通道的标准差；
inplace：是否原地操作
```
例：
~~~
from PIL import Image
from torchvision import transforms 

# 原图
orig_img = Image.open('TP02.jpg') 
display(orig_img)

# 图像转化为Tensor
img_tensor = transforms.ToTensor()(orig_img)

# 标准化

# 定义标准化操作
# norm_oper = transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # RGB三通道，所以mean和std都是3个值
tensor_norm = transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))(img_tensor)

# 再把标准化后的Tensor转化为图像
img_norm = transforms.ToPILImage()(tensor_norm)
display(img_norm)
~~~
##### 多变换操作组合
compose 类
~~~
torchvision.transforms.Compose(transforms)
~~~

### 4.卷积神经网络 Torch.nn

***

### 4.torch.autograd
#### (1).相关概念
- 在训练模型时，通常需要计算函数（通常是损失函数）相对于模型参数的导数或梯度。PyTorch中的autograd包提供了自动微分功能，它能够自动地追踪对张量的所有操作并进行微分。
- 损失函数：损失函数的计算代表了模型预测值与真实标签之间的差异或者不一致程度。在训练模型时，我们希望模型的预测结果能够尽可能地接近真实标签，因此损失函数的计算可以帮助我们衡量模型预测的准确程度。
- 模型参数的导数或梯度：“模型参数的导数”和“梯度”指的是损失函数相对于模型参数的变化率。在训练模型时，我们希望找到使损失函数最小化的模型参数值。为了实现这一目标，我们需要了解损失函数对模型参数的变化敏感程度，即损失函数对模型参数的导数或梯度。通过计算导数或梯度，我们可以知道在当前模型参数值下，沿着哪个方向对参数进行微小调整可以使损失函数减小，从而引导模型向更优的参数值迭代。
#### (2).基本使用
- requires_grad=True  追踪计算历史
- 1、使用 .detach()： 创建一个内容相同但不需要梯度的新张量。
```python
 x_detached = x.detach()
```
2、使用 torch.no_grad()： 在该上下文管理器中执行的所有操作都不会追踪梯度。
```python
with torch.no_grad():y = x + 2
```
- .backward()  计算所有requires_grad=True的张量的梯度
### 