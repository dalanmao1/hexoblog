---
title: ADRC（自抗扰控制器）技术附代码框架
---


自抗扰控制器 ( Auto/Active Disturbances Rejec ion Controller, ADRC)是韩京清学者提出的，是一种继PID控制器后的一种新型的实用的控制技术。
它不是一种独立的技术，可以理解为是对PID现有技术的一种改进，它吸收了PID的精髓，并弥补了PID的一些缺陷。
引用一句话对ADRC的评价：“自抗扰控制器不只是停留在对 PID进行一些修修补补的工作而是一次对经典调节理论的脱胎换骨的革新．”
ADRC主要是由跟踪微分器（TD）、非线性组合、扩张状态观测器（ESO）组成。

> ### PID的一些缺陷

PID是现在大部分工业控制的主流技术，它具有很多优点，像结构简单，适用性强，稳定性好等等。PID的精髓在于用**目标和实际被控对象的输出之间的误差**来消除此误差，这是留给人类的宝贵思想。虽然PID在控制领域的地位无可替代，但是随着对控制品质要求的提高，它的一些“不足之处”就显露出来。

![](https://img-blog.csdnimg.cn/33e65e6202c94f0586295f6ff013b24a.png)

- **原始误差不合理：** PID是用$e=r-y$生成误差，但是目标信号是可以“跳变”的（比如初始时刻，突然从一个目标值到另一个目标值），但输出信号y是有惯性的，不能跳变，让缓慢的遍量y来跟踪快速变化的变量r是不合理的。
- **微分器不好用：** 误差信号$e$的微分信号$\frac{de}{dt}$没有太好的办法，一般用$\frac{de}{dt}=\frac{e(t)-e(t-h)}{h}$来实现。但这样会导致一个问题，当信号被噪声$n(t)$污染之后，就会产生**噪声放大效应**，如有噪声的信号的微分是$\frac{de}{dt}=\frac{e(t)-e(t-h)+n(t)}{h}$，往往时间常数$h$的取值一般都很小，$h$越小$\frac{n(t)}{h}$越大，这个信号因此便被噪声所淹没，失去使用价值。
- **线性组合并不是最好：** PID控制器采用的是误差的现在$e(t)$、误差的过去$\int_{0}^{t}e(t)dt$和误差的未来$\frac{de}{dt}$三者之和相加，但实际表明，这种仅仅相加的线性组合方式并不是最好的，在非线性的领域找到合适的组合是值得探索的。
- **积分带来负面影响：** 如果使用**位置式PID**，也就是$u(t)=k_pe(t)+k_i\int_{0}^{t}e(t)dt+k_d\frac{de}{dt}$，在使用的过程中会产生积分饱和。这个很好理解，当系统存在一个方向的偏差，那么由于积分的存在，$\int_{0}^{t}e(t)dt$越来越大，会使$u(t)$越来越大，从而使$u(t)$到达极限不再改变，这时候积分还是会不断增大。等到系统产生一个反向的偏差，由于积分处于一个非常“大”的状态，即使能减掉反方向的误差，也需要花很长时间脱离饱和区。
  
> ### ADRC

- **安排过渡过程:** 根据控制目标和对象承受能力先安排合适的过渡过程，安排过渡过程的手法在控制工程实 践中已常被采用。如, 升温过程中的“升温曲线”, 热处理过 程中的“温度曲线”等。但是, 大多情形并不利用这些温度曲 线的微分信号。
ADRC提倡的“安排过渡过程”,是同时要给出过渡微分信号的
- **微分信号的提取:** 利用微分近似公式$\frac{de}{dt}=\frac{e(t)-e(t-h)}{h}$会带来噪声放大效应，甚至会淹没噪声。但是如果采用两个惯性环节相减的方式，就能降低噪声放大效应，将原近似公式变为$\frac{de}{dt}=\frac{e(t-h_1)-e(t-h_2)}{h_2-h_1}$,其中的$h_2,h_1$仍是时间常数，这种方式是用二阶动态环节实现微分功能的。
在这个基础上，引入bang-bang最速曲线，实现用最快速度跟踪上输入信号，将“尽可能快”变为“最快”。同时，避免系统在进入稳态时有颤振现象（bang-bang特性），整合出最速综合函数$fhan(x1,x2,r,h)$
公式如下：
$$
\begin{cases}
d=rh^2\\
a_0=hx_2\\
y=x_1+a_0\\
a_1=\sqrt{d(d+8|y|)}\\
a_2=a_0+sign(y)(a_1-d)/2\\
a=(a_0+y)fsg(a,d)-rsign(a)(1-fsg(a,d))\\
fhan=-r(\frac{a}{b})-rsign(a)(1-fsg(a,d))
\end{cases}
$$
其中
$$
fsg(x,d)=(sign(x+d)-sign(x-d))/2
$$

Fhan代码：

```bash
function out=fhan(x1,x2,r,h)
    d=r*h^2;
    a0=h*x2;
    y=x1+a0;
    a1=sqrt(d*(d+8*abs(y)));
    a2=a0+sign(y)*(a1-d)/2;
    a=(a0+y)*fsg(y,d)+a2*(1-fsg(y,d));
    out=-r*(a/d)*fsg(a,d)-r*sign(a)*(1-fsg(a,d));
end
function out=fsg(x,d)
    out=(sign(x+d)-sign(x-d))/2;
end
```

以下的系统我们称为跟**踪微分器(Tracking Differentiator, TD)**

```bash
 %TD-Fhan  input为目标信号，v1跟踪信号，v2微分信号
 v1=v10+h*v20;
 v2=v20+h*fhan(v10-input(i),v20,r,h0);
 ```

- **非线性组合的应用：** 有了“安排过渡过程”和“跟踪微分器”的手段，我们可以用TD产生的误差信号$e_1=v_1-x_1$和误差的微分信号$e_2=v_2-x_2$，再加一个积分信号$e_0=\int_{0}^{t}e_1(t)dt$，就可以设计出PID控制器了，但是大量研究表明，采用非线性组合的方式效果更好。
有如下非线性函数
$$
fal(x,\alpha,\sigma)=
\begin{cases}
\frac{e}{\sigma ^{1-\alpha}},\quad |e|<=\sigma\\
|e|^\alpha sign(e),\quad |e|>\sigma
\end{cases}
$$
等效为：
$$
s==\frac{sign(e+\sigma)-sign(e-\sigma)}{2};\\
\\
fal(x,\alpha,\sigma)=\frac{e}{\sigma ^{1-\alpha}}s+|e|^\alpha sign(e)(1-s)
$$
代码：

```bash
function out=fal(e,alpha,sigma)
     s=(sign(e+sigma)-sign(e-sigma))/2;
     out=s*e/(sigma^(1-alpha))+abs(e)^alpha*sign(e)*(1-s);
end
 ```

- **扩张状态观测器(Extended State Observer, ESO) 与扰动估计补偿：** ESO的设计提出解决了积分带来的负面作用，同时它还可以跟踪系统的原信号并对系统总扰动进行估计预测，可以说ESO时整个ADRC框架的核心部分。
假设有这样的系统：
$$
\begin{cases}
\dot{x_1}=x_2\\
\dot{x_2}=f(x_1,x_2,\omega(t),t)+bu\\
y=x_1
\end{cases}
$$
其中，$\omega(t)$是外部扰动，我们将总扰动f设为：
$$
x_3(t)=f(x_1,x_2,\omega(t),t)
$$
那么，新的系统方程就变成了：
$$
\begin{cases}
\dot{x_1}=x_2\\
\dot{x_2}=x_3+bu\\
\dot{x_3}=\omega_0(t)\\
y=x_1
\end{cases}
$$
注意，这边的$\omega_0(t)$和$\omega(t)$没有太大关系，我们也不知道它等于什么，我们只需要了解用$x_3$构建了一个新的变量——扰动。
线性扩张状态观测器(LESO)方程如下：
$$
\begin{cases}
e=z_1-y\\
\dot{z_1}=z_2-\beta_{01}e\\
\dot{z_2}=z_3-\beta_{02}e+bu\\
\dot{z_3}=-\beta_{03}e\\
\end{cases}
$$
在扰动不是很剧烈的情况下，选取合适的参数$\beta_{01}、\beta_{02}、\beta_{03}$，$z_i$可以很好的跟踪上$x_i$。尽管我们在LESO中没有用到未知函数$\omega_0(t)$, 但是在系统运行过程中它是实实在 在地起作用的, 为了消除其影响, 采用非线性效应, 把方程改造成：
$$
\begin{cases}
e=z_1-y\\
\dot{z_1}=z_2-\beta_{01}e\\
\dot{z_2}=z_3-\beta_{02}fal(e,0.5,h)+bu\\
\dot{z_3}=-\beta_{03}fal(e,0.25,h)\\
\end{cases}
$$
改造以后的扩张观测器的跟踪效果更好, 适应范围更大。非线性的ESO算是真正意义上的“扩张状态观测器”。
代码如下：

```bash
 e=z10-y1; 
 fe=fal(e,0.5,0.005);fe1=fal(e,0.25,0.005);
 z1=z10+eso_h*(z20-beta01*e);
 z2=z20+eso_h*(z30-beta02*fe+b*x1);
 z3=z30+eso_h*(-beta03*fe1);
```

ESO如何消除积分呢？
过程很有意思，利用公式$u=\frac{u_0-z_3}{b_0}$就能代替积分项$k_i\int_{0}^{t}e(t)dt$，感兴趣可以推一下。
