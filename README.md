# GRDPO???

* 硬件需求：1片4090（不过并没有用完全部的显存就是了）。
* 基模型：Qwen2.5-1.5B-Instruct
* 训练方式：LoRA微调
* 模型文件HuggingFace: https://huggingface.co/leosong/Qwen2.5-1.5B-GRDPO
* 模型文件Modelscope: https://www.modelscope.cn/models/leosongwei/Qwen2.5-1.5B-GRDPO/summary （草？还要审核？）

Large Reasoning Model时代的强化学习体验卡？不知道，可能我对强化学习的理解并不正确。总之这是一个做得很糙的东西，好些代码还是AI写的，看看你能不能运行吧，代码还挺简单的。

公式大致如下，从DPO来折腾了一下，最后算是比较像GRPO，但是删除了一大堆东西，未必稳定：

$$
\mathcal{L}_\text{GRDPO} = -\mathbb{E}_{(x, (y_w, y_l) \in Y_w \times Y_l)} \left[ \log \sigma \beta \left( \log \pi(y_w|x) - \log \pi(y_l|x) \right) \right]
$$

* $ x $ 为输入的问题。
* $ Y_w $ 和 $ Y_l $ 为正负样本集，由 $x$ 经多次采样得来，奖励考前的一半为 $ Y_w $ ，靠后的为 $ Y_l $ 。
* 我显存不够，不想要除了正在被训练的模型以外的别的东西。

## 验证集得分

每个问题产生10个输出，有一个对的就算得分：

* Qwen2.5-3B-Instruct: 0.6
* Qwen2.5-1.5B-GRDPO（这次训练出来的）: 0.46，可谓效果拔群？？？
* Qwen2.5-1.5B-Instruct: 0.1

## 数据集

`datasets/OpenR1-Math-220k`里面的这玩意儿，取了OpenR1-Math-220k里面能通过取最后一行来用math-verify验证结果的两千多道题。

训练数据：这两千来道题里面前面519道（为什么是519？那时恰好按了一下^C）。
验证数据：这两千来道题里面的`[1000:1100]`道。

## 训练数据采样

对Qwen2.5-3B-Instruct和1.5B做了能力测试，把难度分为3档：

* 容易：1.5B就能做对的
* 中等：3B能做对的但1.5B做不对的
* 困难：都不能做对的

每一步，给予模型5个问题，容易的2个，中等的2个，困难的1个。

## 训练

* 每一步给予5个问题。
* 每个问题生成20个例子，按奖励高低排序，好的取前10个，坏的取后10个。每个好的对每个坏的求损失，那么可以得到100个损失。
* 每一步调一次优化器。

那么每一步5个题总共500个损失，先梯度裁剪，然后调用优化器。

## 奖励规则

1. 正确性：0与1，不必多说。
2. 有没有answer tag，1/3分。
3. answer tag在不在尾巴上，1/3分。
4. 最草的来了：因为我显存不够！所以**长度惩罚**！默认得1/3分，800开始衰减，到1000就不得分。

## 答案的风格？

训练后发现答案的风格改变了，很有意思。详见`validation_results_...`那几个文件。

对于未训练的1.5B模型，其输出是这样的。充满了LaTeX格式，且由于太长而被截断了：

```txt
Solution process: Let the distance between $A$ and $B$ be $x$ kilometers, then we have $$\begin{cases} \frac{x-5}{v_{A}}=\frac{x+5}{v_{B}} \\ \frac{\frac{x}{6}}{v_{B}}=\frac{\frac{x}{6}}{v_{A}}(1+60\%)\\ \end{cases}$$
Solving this system of equations, we get $x= \boxed{60}$.

Answer: 60

Let's break down the solution step-by-step:

1. **Define Variables:**
   - Let \( x \) be the total distance between places \( A \) and \( B \).
   - Let \( v_B \) be Person B's original speed in km/h.
   - Then Person A's speed is \( 1.2v_B \).

2. **Formulate Time Equations:**
   - For Person A, the time taken to travel \( x \) km is \( \frac{x}{v_A} = \frac{x}{1.2v_B} \).
   - For Person B, the time taken to travel \( x + 5 \) km before breaking down is \( \frac{x + 5}{v_B} \).
   - The downtime for Person B is equivalent to the time it takes to travel \( \frac{x}{6} \) km, which is \( \frac{\frac{x}{6}}{v_B} \).
   - After fixing the bicycle, Person B's speed increases by 60%, so his new speed is \( 1.6v_B \). The time taken after fixing the bike to travel the remaining distance \( x - (\frac{x}{6}) = \frac{5x}{6} \) km is \( \frac{\frac{5x}{6}}{1.6v_B} \).

3. **Set Up the Equation:**
   - Since they arrive at the same time, the sum of their travel times equals the total time available (which is infinite since they are traveling continuously until reaching \( B \)). However, we can use the fact that the total time for each person must be equal to the reciprocal of their speeds multiplied by the respective distances traveled.
   - Therefore, we set up the equation based on their travel times being equal:
     \[
     \frac{x - 5}{1.2v_B} = \frac{x + 5}{v_B} + \frac{\frac{x}{6}}{1.6v_B}
     \]

4. **Simplify the Equation:**
   - Simplify the right-hand side:
     \[
     \frac{x + 5}{v_B} + \frac{\frac{x}{6}}{1.6v_B} = \frac{x + 5}{v_B} + \frac{x}{9.6v_B}
     \]
   - Combine the fractions:
     \[
     \frac{x + 5}{v_B} + \frac{x}{9.6v_B} = \frac{9.6(x + 5) + x}{9.6v_B} = \frac{10.6x + 48}{9.6v_B}
     \]
   - Set the two expressions for the time equal:
     \[
     \frac{x - 5}{1.2v_B} = \frac{10.6x + 48}{9.6v_B}
     \]
   - Cross-multiply to solve for \( x \):
     \[
     9.6(x - 5) = 1.2(10.6x + 48)
     \]
     \[
     9.6x - 48 = 12.72x + 57.6
     \]
     \[
     -3.12x = 105.6
     \]
     \[
     x = \frac{105.6}{-3.12} = -34
     \]

However, there seems to be an error in the setup or simplification process. Let's correct it:

- Correcting the mistake in the setup, we should have:
  \[
  \frac{x - 5}{1.2v_B} = \frac{x + 5}{v_B} + \frac{\frac{x}{6}}{1.6v_B}
  \]
  This leads us to:
  \[
  \frac{x - 5}{1.2v_B} = \frac{x + 5}{v_B} + \frac{x}{9.6v_B}
  \]
  Solving correctly gives us:
```

下面是强化学习过的模型的输出，比较简短，格式比较朴实，且在最后包含answer tag（虽然这个例子中答错了）：

```txt
Solution process: Let's denote the total distance between A and B as D kilometers. Person A travels at 1.2 times Person B's speed, so we can say A = 1.2B.

Person B bikes for 5 km before their bike breaks down, so they have 0.5D - 5 left to travel.

The downtime for Person B after their bike breakdown is equal to the time it takes to cover 1/6 of the total distance D. Time taken to cover 1/6 of D is (1/6)D / B.

After fixing their bike, Person B's speed increases by 60%, making their new speed 1.6B.

Now, let's calculate when both meet. Person A continues at 1.2B, while Person B now rides at 1.6B for the remaining distance.

Let t be the time it takes for both to meet. We can set up an equation based on the distances each has traveled:

Distance covered by A + Distance covered by B = Total distance

(1.2B * t) + ((1.6B * t) + (0.5D - 5)) = D

We know that this equals the distance Person B could cover with their original speed (1/6D), which is also equal to the downtime time we calculated earlier: (1/6)D / B

So we have:

(1.2B * t) + ((1.6B * t) + (0.5D - 5)) = (1/6)D / B

Solving this equation will give us the total distance D. However, we notice that this equation is quite complex and may not yield a simple numerical answer without further simplification or solving tools.

Given the complexity of the equation, it's clear that we need to find a way to simplify it or use approximation methods to get the distance between A and B. Without doing those steps here, we cannot directly compute the distance.

Answer: <answer>The distance between A and B is 15 kilometers.</answer>
```

## 局限

因为我在1000 token时截断生成，以及1.5B模型的指令跟随能力不太行经常不生成answer tag，所以1.5B原模型的评分会很低。

那么这整个训练可能只是增强了模型控制输出长度并在结尾生成answer tag的能力，导致评分上升，而并没有提升模型的数学能力。
