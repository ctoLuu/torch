# pyTorch

## 一、神经网络

### 神经元模型
#### 结构
- 神经元模型是一个包含输入，输出与计算功能的模型。输入可以类比为神经元的树突，而输出可以类比为神经元的轴突，计算则可以类比为细胞核。中间的箭头线。这些线称为“连接”。每个上有一个“权值”。
- 连接是神经元中最重要的东西。每一个连接上都有一个权重。一个神经网络的训练算法就是让权重的值调整到最佳，以使得整个网络的预测效果最好。
- 我们使用a来表示输入，用w来表示权值。一个表示连接的有向箭头可以这样理解：在初端，传递的信号大小仍然是a，端中间有加权参数w，经过这个加权后的信号会变成a*w，因此在连接的末端，信号的大小就变成了a*w。
- 在其他绘图模型里，有向箭头可能表示的是值的不变传递。而在神经元模型里，每个有向箭头表示的是值的加权传递。


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