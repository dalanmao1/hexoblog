---
title: BP神经网络
---


> ## 1. 人工神经网络

### 1.1 原理

人工神经网络(Artificial Neural Networks，ANN)也简称为神经网络(NN)，它是一种模仿动物神经网络行为特征，进行分布式并行信息处理的算法数学模型。这种网络依靠系统的复杂程度，通过调整内部大量结点之间相互连接的关系，从而达到处理信息的目的。

神经网络首先要以一定的学习准则进行学习，以

决定神经网络模型性能的三大要素为： 神经元(信息处理单元)的特性； 神经元之间相互连接的形式——拓扑结构； 为适应环境而改善性能的学习规则。

### 1.2 特点

人工神经网络是有**自适应**能力的，它可以在训练或者学习过程中改变自身的权重，来适应环境要求。

- **训练方式**
  - 有监督：利用给定的样本进行分类或者模仿。
  - 无监督：规定学习规则，学习的内容随系统所处环境而异，系统可以自发的发现环境的特征，类似人脑的的功能。
- **泛化能力** 对那些没有训练过的样本，有很好的预测能力和控制能力，特别是存在一些噪声的样本。**比如**，公式$y=2x^2$的x和y的值作为样本，假设样本只取x=[0,100],y=x;神经网络训练完之后得到近似公式$y=2x^2$的数学模型,即使没有训练过x=999的这个样本，但是仍可以神经网络近似的数学模型知道y的值是多少。
- **模拟/辨识能力：** 当设计者很清楚的知道系统时，可以用数值分析，微积分等数学工具建立，但是当系统很复杂的时候，或者系统未知，信息量很少，建立精确的数学模型就很困难。神经网络的非线性映射能力就表现出优势，因为它不需要对系统有很透彻的了解，**只需要知道系统输入与输出的映射关系即可近似得到一个数学模型**。

> ## 2. BP神经网络

BP神经网络是一种典型的非线性算法。BP神经网络由**输入层、输出层和之间若干层（一层或多层）隐含层**构成，每一层可以有若干个节点。层与层之间节点的连接状态通过权重来体现。

![](https://img-blog.csdnimg.cn/1c40769c8cc94820a1d15aadf765a0a3.png#pic_center)

### 2.1 感知器

上图可以看到BP神经网络时由一个个节点和线连接起来组成的网络，组成这个网络的单元就叫**感知器**。

![](https://img-blog.csdnimg.cn/f2233f14be9542bdac84b8076e71f00b.png#pic_center)

感知器的结构由输入项$x_i$、权重$w_i$、偏置$\theta$、激活函数$f(\cdot)$、输出$y$组成。其中偏置和激活函数可能不太好理解为什么要把它们放在里面。

首先，偏置（有些文献叫阈值），用一张图来解释为什么要用它，神经网络的作用是对样本数据进行学习然后得到一个近似的模型，细想本质就是寻找规律然后对数据进行分类。

![](https://img-blog.csdnimg.cn/6bbff835924b401db44b0f6707f922a2.png#pic_center=60x60)


为什么采用激活函数？
第一点，假设每个感知器的表达为$y=ax+b$，如果没有激活函数，当x的值也就是输入很大时，经过层层的计算，那这个数值将会非常非常大，激活函数的作用就是将感知器的输出限制在一个固定区间，比如[0,1]。
第二点，按照感知器的输出来看，它是线性的，就算再经过下一层的网络，它仍然是线性的，但是神经网络是需要有非线性的能力的，激活函数正好能赋予它这个能力。


sigmoid激活函数的特点：

- 上下有界（relu是个特例）
- 连续光滑，连续可微（不满足这一条后续就无法做偏导）
- 该函数的中区高增益部分解决了小信号需要放大的问题，两侧的低增益区适合处理大信号。

下图是sigmoid激活函数曲线，类似的有很多，就不一一列举了。
$$
\delta(x)=\frac{1}{1+e^{-x}}
$$

![](https://img-blog.csdnimg.cn/35d07c8636834b89abd0b5bfd5c59b29.png#pic_center=60x60)

> ## 3. 正向传播和反向传播

BP神经网络的学习过程如下：

![](https://img-blog.csdnimg.cn/151c619cf6c1496584eded648678b29e.png#pic_center=60x60)

### 3.1 正向传播

输入样本从输入层传入，经过各隐层处理之后，传向输出层。

![](https://img-blog.csdnimg.cn/7c30c24f60224ad9be18ccea46985323.png#pic_center=60x60)

输入层有n个神经元组成，
输入层的输出：$x_i(i=1,2,...,n)$

隐藏层有q个神经元组成,输入层和隐藏层的权值为$v_{ki}(i=1,2,...,n;k=1,2,...,q)$
隐藏层输入（含阈值$\theta$）：
$$
S_k=\sum_{i=1}^{n}v_{ki}\cdot x_i
$$
隐藏层输出：
$$
z_k=f(S_k)
$$

输出层由m个神经元组成，$y_j(j=1,2,...,m)$为该层的输出
隐藏层和输出层的权值为$w_{jk}(k=1,2,...,q;j=1,2,...,m)$
输出层输入：
$$
S_j=\sum_{k=1}^{q}w_{jk}\cdot z_k
$$
输出层输出：
$$
y_j=f(S_j)
$$

### 3.2 反向传播

正向传播是输入样本从输入层传入，经过各隐层处理之后，传向输出层。若输出层的实际输出与期望的输出不符合，则开始**反向传播**：将误差用某种形式通过隐藏层逐层反传，并将误差分摊给各层的神经元，从而获得各层神经元的误差信号，用于修正各个神经元的权值。
神经元的正向传播和反向传播是周而复始进行的，一直进行到误差减少到可以接受的程度为止，神经网络学习的过程就是不断修正权值的过程。

#### 3.2.1 梯度下降

当一个人站在山顶或者半山腰，想要以最快的速度到达山脚，那么哪条路最快呢？
是每一步都下降最快的路，变化最快。在数学上，用梯度的概念来描述。

下面用梯度下降法求函数的最小值问题：
![](https://img-blog.csdnimg.cn/02340bf852c3483498ee3a55ceceda3e.png#pic_center=60x60)

![](https://img-blog.csdnimg.cn/58e158819a4f4b8ab25356c2af810b2d.png#pic_center=60x60)

梯度是一个矢量，表示某一函数在该点处的方向导数沿着该方向取得最大值，只需要不断的求偏导进行迭代就可以以最快速度到达最小值。
这边的学习率也就是步长，如果取太长，会在最小值左右来回震荡；如果的取的太小，会增加学习时长，一般在[0,1]之间。

#### 3.2.2 公式推导

期望输出用符号t表示，误差的形式一般采用：
$$
E=\frac{1}{2}\sum_{j=1}^m(y_j-t_j)^2
$$

根据梯度下降原理，对每个权值的修正方向为E的函数梯度的反方向,确保$\omega_{ij}$为负值。

$$
\triangle \omega_{ij}=-\eta \frac{\partial E}{\partial \omega_{ij}}
$$

对于输出层
求
$$
\triangle \omega_{jk}=-\eta \frac{\partial E}{\partial \omega_{jk}}
$$
根据链式法则，可以把它转化为两式相乘，$E$对输出层的输入$S_j$求偏导；$S_j$对权值求偏导
$$
\frac{\partial E}{\partial \omega_{jk}}=\frac{\partial E}{\partial S_j} \frac{\partial S_j}{\partial \omega_{jk}}
$$
进一步展开：
$$
\frac{\partial E}{\partial S_j}=(y_j-t_j)f'(S_j)
$$
还记得输出层的输入吗，$$S _j=\sum_{k=1}^{q}w_{jk}\cdot z_k$$
所以得到
$$
\frac{\partial S_j}{\partial \omega_{ij}}=z_k
$$
整理上述，隐藏层和输出层的权值变化量可表示为：
$$
\triangle \omega_{jk}=\eta(t_j-y_j)f'(S_j)z_k
$$
其中$f'(S_j)$是激活函数的导数，不同的激活函数有着不一样的求导函数。

---------------------------------------------------
隐藏层相对麻烦一点，因为隐藏层的输入为$S_{k}$，表达式为：
$$
\triangle v_{ki}=-\eta \frac{\partial E}{\partial v_{ki}}=-\eta \frac{\partial E}{\partial S_{k}}\frac{\partial S_{k}}{\partial v_{ki}}
$$
先看前半部分，因为隐藏层的输出为$z_k=f(S_k)$**（如果看不明白，最好把正向传播的公式抄下来对着看）**，可以展开为：
$$
\left\{
\begin{aligned}
&\frac{\partial E}{\partial S_{k}}=\frac{\partial E}{\partial z_{k}} \frac{\partial z_{k}}{\partial S_{k}}\\
&\frac{\partial S_{k}}{\partial v_{ki}}=x_i
\end{aligned}\right.
$$

进一步展开：
$$
\left\{
\begin{aligned}
&\frac{\partial E}{\partial z_{k}}=\sum_{j=1}^{m}(y_j-t_j)\frac{\partial y_{j}}{\partial z_{k}}\\
&\frac{\partial z_k}{\partial S_{k}}=f_z'(S_k)
\end{aligned}\right.
$$


其中$f_z'(S_k)$是隐藏层的激活函数的导数，$y_i$是输出层的输出，继续展开：
$$
\frac{\partial y_{j}}{\partial z_{k}}=\frac{\partial y_{j}}{\partial S_{j}}\frac{\partial S_{j}}{\partial z_{k}}=f'(S_j)\frac{\partial S_{j}}{\partial z_{k}}
$$
其中$f'(S_j)$为输出层的激活函数，和上面的$f_z'(S_k)$不要弄混
$$
\frac{\partial S_{j}}{\partial z_{k}}=\omega_{jk}
$$

又回到最初的公式，整理上面的公式：
$$
\triangle v_{ki}=\eta x_if_z'(S_k)(\sum_{j=1}^{m}(t_j-y_j)f'(S_j)\omega_{jk})
$$
最终，得到隐藏层和输出层的权值公式为：
$$
\left\{
\begin{aligned}
&\triangle \omega_{jk}=\eta(t_j-y_j)f'(S_j)z_k\\
&\triangle v_{ki}=\eta x_if_z'(S_k)(\sum_{j=1}^{m}(y_j-t_j)f'(S_j)\omega_{jk})
\end{aligned}\right.
$$
两个权值有公共部分，可简化为：
$$
\left\{
\begin{aligned}
&\delta=(t_j-y_j)f'(S_j)\\
&\triangle \omega_{jk}=\eta\delta z_k\\
&\triangle v_{ki}=\eta x_if_z'(S_k)(\sum_{j=1}^{m}\delta\omega_{jk})
\end{aligned}\right.
$$
对公式中的符号再说明一下，$t_j$表示期望输出；$y_j$表示实际的输出，一般用输出层输出表示；$\eta$表示学习率；$z_k$表示隐藏层的输出；$x_i$表示输入层的输入（或输出）；$\omega_{jk}$表示隐藏层和输出层的权值；$S_{j}$和$S_{k}$分别为隐藏层和输出层的输入；$f_z'(\cdot)$和$f'(\cdot)$为隐藏层和输出层的激活函数。

#### 3.2.3 激活函数

对于激活函数的选择，有很多方式，可以选择同一个激活函数，那么$f(\cdot)=f_z(\cdot)$，以下是激活函数及其的导数形式

激活函数     | 导数形式
------- | -----
$sigmoid(x)=\frac{1}{1+e^{-x}}$  | $sigmoid'(x)=\frac{1}{1+e^{-x}}(1-\frac{1}{1+e^{-x}})$ 
$tanh(x)=\frac{e^{x}-e^{-x}}{e^{x}+e^{-x}}$  | $tanh'(x)=1-tanh^2(x)$ 
$relu(x)=\begin{cases}x\quad x\geqslant 0 \\ 0\quad x< 0 \end{cases}$  | $relu'(x)=\begin{cases}1\quad x\geqslant 0 \\ 0\quad x< 0 \end{cases}$

### 3.3 权值更新

神经网络的修正和更新主要是对权值进行更新和修正，我们已经知道了每一层的$\triangle \omega$的具体公式，用下面公式进行更新：
$$
\omega_{ij}\leftarrow\omega_{ij}+\triangle \omega_{ij}
$$

#### 3.3.1 神经网络存在的问题

神经网络存在一些问题

- 局部最小：在某些初始条件的权值下，会陷入局部最小的问题。有时候局部最小离全局最小值较远的情况下会导致学习失败。
- 学习率选取问题：如果取太长，可能会在最小值左右来回震荡；如果的取的太小，会大大增加学习时长。
- 隐藏层的节点个数的确定缺少完整的理论指导。

#### 3.3.2 添加动量项

将权值修正公式变为
$$
\triangle \omega(n)=-\eta \frac{\partial E(\omega)}{\partial \omega(n)}+\alpha \triangle \omega(n-1)
$$
$$
\omega_{ij}\leftarrow\omega_{ij}+\triangle \omega_{ij}
$$

添加动量因子$\alpha$（$0<\alpha <1$）的本质是改变学习率，一开始误差较大时可以提高收敛速度，并抑制训练过程中的震荡，起到了缓冲平滑的作用。
此外，如果网络的训练已进入了误差曲面的平坦区域，那么误差将变化很小，权值的调整幅度也将相应的变得非常小，也就是说$\triangle \omega(n-1)$近似为$\triangle \omega(n)$，公式可以看作：
$$
\triangle \omega(n)=-\frac{\eta}{1-\alpha}\frac{\partial E(\omega)}{\partial \omega(n)}
$$
从而有利于快速脱离饱和区

> ## 程序

用神经网络去逼近函数$y=\frac{3}{10}x^2-2x-1$

```matlab
%BP神经网络逼近实例
clc;
close all;
clear all;
xite=0.050; % 学习速率
alfa=0.05; % 动量因子
%产生随机信号w1,w2
w2=rands(6,1);

w2_1=w2;w2_2=w2_1;
w1=rands(2,6);
w1_1=w1;w1_2=w1;
dw1=0*w1;%BP网络逼近初始化
x=[0,0]';
u_1=0;
y_1=0;
I=[0,0,0,0,0,0]';
Iout=[0,0,0,0,0,0]';
FI=[0,0,0,0,0,0]';
maxtime=500;


%BP神经网络逼近开始
for k=1:1:maxtime
    %目标函数
    X=k/100;
    y(k)=0.3*X*X-2*X-1;
    u(k)=X;
   x(1)=u(k);
    x(2)=y(k);
    for j=1:6
        I(j)=x'*w1(:,j);
        Iout(j)=1/(1+exp(-I(j)));
    end
    yn(k)=w2'*Iout; % 神经网络的输出
    e(k)=y(k)-yn(k); % 构造误差性能指标函数
    w2=w2_1+(xite*e(k))*Iout+alfa*(w2_1-w2_2); % 计算权值w2
    for j=1:6
        FI(j)=exp(-I(j))/(1+exp(-I(j)))^2;
    end
    for i=1:2
        for j=1:6
            dw1(i,j)=e(k)*xite*FI(j)*w2(j)*x(i);
        end
    end
    w1=w1_1+dw1+alfa*(w1_1-w1_2); % 计算权值w1
    %计算jacobian阵
    yu=0;
    for j=1:6
        yu=yu+w2(j)*w1(1,j)*FI(j);
    end
    dyu(k)=yu;
     
    w1_2=w1_1;
    w1_1=w1;
    w2_2=w2_1;
    w2_1=w2;
    u_1=u(k);
    y_1=y(k);
end
time=1:1:maxtime;
figure(1);
plot(time,y,'r',time,yn,'b--','linewidth',2.5);
xlabel('times');
ylabel('y and yn');
legend('y','yn');
grid;
title('BP神经网络系统逼近');
figure(2);
plot(time,e,'r');
xlabel('times');
ylabel('error');
grid;
title('误差曲线')

```

## 参考文献

[1]冷雨泉等. 《机器学习入门到实战MATLAB实践应用》[J]. 清华大学出版社, 2019
[2]刘益民. 基于改进BP神经网络的PID控制方法的研究[D].中国科学院研究生院（西安光学精密机械研究所）,2007.