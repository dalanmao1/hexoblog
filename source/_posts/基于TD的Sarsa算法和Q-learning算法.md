---
title: 基于TD的Sarsa算法和Q-learning算法
---
TD方法可以划分为两类：在线策略和离线策略（或者叫同轨策略和离轨策略），先要学习的是动作价值函数而不是状态价值函数。
>## Sarsa算法
对于同轨策略，必须对所有的状态s和该状态下的所有动作a，估计出在当前策略下对应的$q_\pi(s,a)$，其实和上一节中的TD学习的公式同理，定义为：
$$
Q(S_t,A_t)\leftarrow Q(S_t,A_t)+\alpha[R_{t+1}+\gamma Q(S_{t+1},A_{t+1})-Q(S_t,A_t)]
$$
对应的TD误差为：
$$
\delta_t=R_{t+1}+\gamma Q(S_{t+1},A_{t+1})-Q(S_t,A_t)
$$
算法的过程可表示如下：
![](https://img-blog.csdnimg.cn/429fbb9b86c243c394023a6bf7dcf9c8.png#pic_center)

>## Q 学习算法
Q学习是一种离轨策略下的时序差分控制算法，如果说Sarsa算法是以当前的动作价值函数$q_\pi(s,a)$作为学习目标，那么Q学习就是以最优动作价值函数$q_*(s,a)$作为学习目标，其定义为：
$$
Q(S_t,A_t)\leftarrow Q(S_t,A_t)+\alpha[R_{t+1}+\gamma \underset {a}{max}Q(S_{t+1},a)-Q(S_t,A_t)]
$$
注意，上式很容易搞错的一点就是为什么是$Q(S_{t+1},a)$而不是$Q(S_{t+1},A_{t+1})$，因为Q学习会去找下一个状态下的最大Q值的一个动作，而后者指的是下一状态下的**一个**动作，这个动作并不具有最优动作价值函数，这也是和Sarsa算法最大的一个区别，不区分这点的话，编程可能会把Q学习写成Sarsa。
算法的过程可表示如下：
![](https://img-blog.csdnimg.cn/0d3084a5e86b4ac78bdeb35cce8fe2e6.png#pic_center)

>## 基于Q学习的PID控制器设计（磁浮车）
- **step1：设计Q表**
Q表是**状态-动作**二元组，用什么表示状态？用什么表示动作？
&emsp;&emsp;状态是需要观测到的，并根据状态来给出动作，智能体获得奖励，所以我们用整个系统的输出也就是间隙信号作为状态。需要注意的是！磁浮车的间隙信号是一个连续信号（8mm->4mm），Q学习算法不能处理连续信号，对该连续信号进行**离散化**处理。把连续信号分为N个区间，每一个区间对应一种状态，同一种状态有着相同的处理。比如，将8mm->4mm分为下表所示的5个状态，对应的每一种状态都会有一组控制器参数控制，从8mm到4mm之间的过程是由多组参数共同作用的。
&emsp;&emsp;关于动作的设计，我查阅了许多文献，都没有具体说明Q表应该怎么设计，但是引入Q学习的目的是为了参数寻优，动作的设计一定和控制器参数相关，最简单的方法就是将参数设为动作。


PID控制器有三个参数，所以需要设计三个q表，以kp表为例，状态为间隙的区间，动作为参数具体数值，表内的值为Q值，可以理解为累积奖励值。
| 状态/动作 | 1   | 2   | ... | 5000 |
| --------- | --- | --- | --- | ---- |
| (8,7]     | 0   | 0   | 0   | 0    |
| (7,6]     | 0   | 0   | 0   | 0    |
| (6,5]     | 0   | 0   | 0   | 0    |
| (5,4]     | 0   | 0   | 0   | 0    |
| (4,0)     | 0   | 0   | 0   | 0    |


- **step2：设计策略**
策略是状态到动作的映射，根据状态选择合适的策略。
&emsp;&emsp;强化学习本质上是一种试错的学习方式，一开始智能体并不知道哪一个动作是“好的”，哪一组参数是优秀的，能缩小实际输出与目标输出的距离起到调控作用。在设计策略时，需要设计两种策略，刚开始智能体不知道对于当前的状态来说，哪一个动作时最优的，所以刚开始采用**随机策略**选择动作。比如，kp的值范围是（3000，5000），一开始智能体会在该区间内随机选择一个数。
&emsp;&emsp;随着一轮一轮学习过后，Q表中的值就会发生变化，每次智能体选择一个参数后，都会获得相应的奖励，这个参数的效果越好，获得的奖励越多，那么最后每个参数获得的累积奖励是不一样的。学习到一定次数后，Q表逐渐趋于稳定，智能体用**贪心策略**去选择动作，每次选择累积奖励最多的那个参数，因为控制效果越好获得的奖励越多，累积奖励也就越多。


- **step3：设计奖励机制**
如何判断效果好不好？如何根据效果去给予奖励？
&emsp;&emsp;PID的最终目的是将系统的输出与目标值的误差调为0。针对磁悬浮系统的情况将奖励分为三种情况：调节后输出间隙趋于目标间隙，调节后输出间隙在目标间隙附近，调节后输出间隙远离目标间隙。
&emsp;&emsp;①调节后输出间隙趋于目标间隙：若输出间隙与目标间隙的误差绝对值|e(t)|大于误差最小值，但是小于上一次的误差绝对值|e(t-1)|，意为此次的调节有效，奖励值为误差绝对值的倒数再乘以一个系数k_r，即与误差最小值的偏差越小，获得的奖励越多，这里的k_r取目标间隙的5%；
&emsp;&emsp;②节后输出间隙在目标间隙附近：若输出间隙与目标间隙的误差绝对值|e(t)|小于误差最小值时，意为此次调节效果显著，奖励值为100。
&emsp;&emsp;③调节后输出间隙远离目标间隙：若输出间隙与目标间隙的误差绝对值|e(t)|大于上一次的误差绝对值|e(t-1)|，意为此次的调节无效，奖励值为0。
- **step4：学习样本区间**
&emsp;&emsp;上面提到，强化学习是一种试错的学习方式，一开始采用随机策略选择的参数是在样本集合上的，而磁浮车系统是经不起试错的，因为一旦选择了一个“不好”的参数，就会导致最后的间隙超过8mm，也就是车体吸死在轨道上，这是不能出现的情况。
&emsp;&emsp;样本区间太大，不仅会导致实际车体在训练中上下抖动，甚至吸死，而且还会使学习的时间变得很长。设计的话可以结合劳斯判据和经验来决定区间，这里设计kp的范围是[3000,6000]，ki的范围是[4,20],kd的范围是[30,80]。
- **step5：学习过程**
	<font face="楷体" >
    - 初始化三个Q表，Q_i (s,a)=0，i=1,2,3；
	- 初始化学习率α_i,i=1,2,3；
	- 初始化折扣因子γ；
	- 初始化ε－greedy策略的ε；
	- while ( episode＜maxepisode ) 循环执行；
	- &emsp;执行ε－greedy策略，ε衰变；
	- &emsp;for ( t=1:maxtime)循环执行；
	- &emsp;&emsp;将间隙信号x ( t )和间隙微分信号x ̇(t)离散化，得到n_1 (t)，n_2 (t)；
	- &emsp;&emsp;遵循ε－greedy策略，根据n_1 (t)，n_2 (t)选择动作A；
	- &emsp;&emsp;通过PID控制，得到完整的输出；
	- &emsp;&emsp;观察新状态
	- &emsp;&emsp;得到奖励R_i，i=1,2,3；
	- &emsp;&emsp;将新状态离散化，得到n_1 (t+1)，n_2 (t+1)；
	- &emsp;&emsp;更新Q_i (s,a)的学习率α_i，i=1,2,3；
	- &emsp;&emsp;根据公式（11）更新Q_i (s,a)，i=1,2,3；
	- &emsp;end
    - end
</font>


- **遇到的一些问题和解决方式**
  - 1. 离散化的时候值得注意的是，因为磁浮车最后会稳定在4mm左右，设计的区间最好是像[3.5,4.5]这种，如果设计成[3,4],[4,5]这样的话，容易在两个区间反复横跳，会出现一种状态有两组参数控制的现象。
  - 2. 仿真线条发散。样本设计区间太大，在学习次数不变的情况下，找不到合适的参数。或者从随机策略到贪心策略的过渡太靠前了，这边是在前60%的学习次数中选择随机策略用于试错找到合适的动作，后40%选用贪心。
  - 3. 每次学习完之后参数都不一样。Q学习算法会出现局部最优但不是全局最优的情况，而且和样本大小和学习次数也有关系。
  - 4. 学习时间太长了。适当缩小学习样本和减少学习次数，添加一个终止状态$S_T$,最终控制到的状态是4mm左右的时刻，一旦到了这个状态就可以停止这一轮的学习了，减少了在这个状态下学习的时间可以大大减少总学习时间。根据输出的图可以看到，用大约maxtime*1/3的时间就能从8mm调控到4mm，所以可以将maxtime（完整一轮pid控制的时间）变成原来的1/3或者1/2，即设定一个终止状态，一到终止状态就停止当前的学习。
  - 5. 奖励机制的问题。正奖励大致上分为两种，一种就是能调控到目标值左右，记为奖励R；另一种是不能调控到目标值左右，但是能缩小与目标值的差距，记为奖励r。R应该要远大于r，在R不可取的情况下才考虑r的。
  - 6. 因为强化学习不依赖于实际的数学模型，所以不需要知道磁浮车的模型。
>## 基于Q学习的ADRC控制器设计（磁浮车）
