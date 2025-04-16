# README
实现`torch.nn.init`包中的初始化方法

## 参考资料
[torch.nn.init](https://pytorch.org/docs/stable/nn.init.html)
[Self-Normalizing Neural Networks](https://arxiv.org/pdf/1706.02515)

## Self-Normalizing Neural Networks
《Self-Normalizing Neural Networks》是由 Günter Klambauer 等人在 2017 年提出的一篇深度学习论文，核心思想是设计一种网络结构和激活函数，使得网络在前向传播中可以自动保持激活值的均值为0、方差为1的“标准化”状态，从而不依赖于 Batch Normalization 或其他归一化技术。这种机制有助于解决深层网络中的梯度消失/爆炸问题。

**SNN（Self-Normalizing Neural Network）定义**  
SNN 的目标是使神经网络的每一层都能自动保持输出分布稳定，也就是:
- 均值接近 0
- 方差接近 1    

这样做可以减少梯度消失或爆炸，提升训练速度和稳定性。

**SELU 激活函数**   
SNN 的关键之一是引入了 SELU（Scaled Exponential Linear Unit） 激活函数：

$$
\text{SELU}(x) = \lambda \begin{cases}
x & \text{if } x > 0 \\
\alpha (e^x - 1) & \text{if } x \leq 0
\end{cases}
$$

其中参数为：
- $\lambda \approx 1.0507$
- $\alpha \approx 1.6733$

相比 ReLU、tanh 等，SELU 本身具备“自归一化”特性。


**前提条件**    
SELU 要实现效果，必须满足几个假设条件： 
- 网络是全连接前馈网络
- 初始化权重使用LeCun Normal
- 使用 SELU 激活函数
- 输入数据是归一化的（均值0、方差1）

**理论分析**    
作者从数学上分析了当多个 SELU 层串联时，输出均值和方差如何逐渐收敛到一个固定点（fixed point）：     
- 网络越深，输出的均值和方差会自动趋向一个稳定值（self-normalizing effect）。


**实验效果**
- 在多个数据集上（如 MNIST, CIFAR 等），SNN + SELU 的效果和训练速度都优于传统的 ReLU + BatchNorm 网络。
- 在不使用 batch norm 的情况下，也能训练很深的神经网络。

## xavier和kaiming初始化
**xavier初始化**
- 适用于： sigmoid / tanh / linear 激活函数（它们对方差比较敏感）
- 不推荐： ReLU/LeakyReLU（因为它们会将一部分输出置零，方差不守恒）

**kaiming初始化**
适用于ReLU和LeakyReLU