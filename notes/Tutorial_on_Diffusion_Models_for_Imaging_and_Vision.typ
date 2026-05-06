// 封面变量，统一管理封面内容
#let cover = (
  title: "大二下学习笔记",
  subtitle: "",
  author_line: [#link("https://zhangqiyu.me")[Qiyu Zhang] 编著],
  logo: "myimage.jpg", // 相对或绝对路径
  publisher: [#link("https://zhangqiyu.me")[https://zhangqiyu.me]], 
  date: "2025年11月",
)


//设置全局字体样式与大小
#set text(
  font:("Consolas","Simsun"),
  size:12pt,
  // lang:"zh",
  // region:"cn"
)

//设置首行缩进与行距
#set par(
  first-line-indent: (amount: 0em, all: false),
  justify: true,//两端对齐
  leading: 0.8em,
)

//#show selector: it => { ... }：为匹配到的节点定义渲染器，it 是原始的标题内容（保留它可使内容按原样插入）

#show heading.where(level:1): it => {
  v(0em)
  set align(center)//居中
  set text(
    font:("STKaiti"),
    size:20pt,
    weight:"bold",//默认加粗
  )
  it
  v(2em)
}

#show heading.where(level:2): it => {
  v(0em)
  set align(left)//居中
  set text(
    font:("SimHei"),
    size:18pt,
    weight:"bold",//默认加粗
  )
  it
  v(1em)
}

#show heading.where(level:3): it => {
  v(2em)
  set align(center)
  set text(
    font:("SimHei"),
    size:16pt,
    weight:"bold",//默认加粗
  )
  it
  v(1em)
}

#show heading.where(level:4): it => {
  v(0em)
  set align(left)//居中
  set text(
    font:("SimHei"),
    size:15pt,
    weight:"bold",//默认加粗
  )
  it
  v(2em)
}


#show link: it => {
  set text(blue)
  it
}
// ==================== 封面页 ====================
#page(
  margin: 0cm,
  header: none,
  footer: none,
  numbering: none,
)[
  #align(center)[
    #v(3cm)
    #text(size: 36pt, weight: "bold")[#cover.title]
    #v(1cm)
    #text(size: 20pt)[#cover.subtitle]
    #v(4cm)
    #text(size: 16pt, font:"Consolas")[#cover.author_line]
    #v(5cm)
    #image("../myimage.jpg", width: 5cm)  // 图片参数为表达式，保持不变
    #v(1cm)
    #text(size: 14pt, font:"Consolas")[#cover.publisher]
    #v(0.5cm)
    #text(size: 14pt, font:"STKaiti")[#cover.date]
  ]
]


#set page(numbering: none)
#outline(
  depth:none,//显示目录的深度
  indent:2em,//目录缩进
  target:heading.where(outlined:true),//目录中应包含的元素类型，默认为heading.where(outlined:true)
  title:[目录],//默认为一级标题
)


#set page(
  paper:"a4",
  margin:(top: 2cm, bottom: 2cm, left: 2cm, right: 2cm),
  numbering: "1",
  number-align: center,

  footer: context {
    align(center)[
      #v(0em)
      #counter(page).display()
    ]
  }
)


#pagebreak()
#set page(numbering: "1")
#counter(page).update(1)//重置页码

#show strong: set text(purple, weight: "bold")
//#v(3em)

#let at(con) = {
  text(font: "STKaiti", size: 12pt, weight: "bold")[#con]
  v(1em)
}

#let the(con)={
  text(red)[#con]
}

#import "@preview/fletcher:0.5.8": diagram, node, edge

= 一·基础知识：变分自编码器（VAE）

== 1.1 VAE设置

#h(2em)假如拿到10w条冰淇淋日销量记录$D={x_i}$，我们相信存在某些隐藏因素（天气、情绪、季节…）决定了销量，但我们观测不到它们，那么可以通过VAE来从数据中学习一个可以刻画数据真实概率密度$p(x)$的生成模型，并生成新的、与真实数据同分布的冰淇淋销量数据。

#h(2em)我们假设隐变量（天气、季节等因素）为$z$（可以是个多维变量）。作为一个生成模型，VAE的核心组件之一，解码器。它的任务就是根据给定的隐藏因素$z$，生成一个对应的冰淇淋销量$x$。因为销量是实数，最简单自然的选择就是用一个高斯分布来描述：$ p_(theta)(x|z)=cal(N)(x|mu_(theta)(z),sigma^2) $在这里，$mu_(theta)(z)$是个神经网络，输入是隐变量$z$，输出是一个实数，代表生成的冰淇淋销量的均值；$sigma^2$是一个固定的超参数，代表生成时的噪声大小。具体来说，如果你告诉我今天的隐藏因素向量$z$，解码器会先通过神经网络算出一个预测的数字$mu_(theta)(z)$，然后再以这个数字为中心，加上一点随机扰动$sigma$，最终生成一个具体销量$x$。比如说，某个$z$代表“阴冷潮湿的冬天”，解码器很可能预测一个很低的销量（比如-2），但因为有随机性，实际生成的销量可能在-3到-1之间波动。

#h(2em)但是，仅仅有解码器是不够的，因为我们并不知道每个销量$x$背后对应的$z$到底是什么。理论上我们应该用贝叶斯公式计算后验分布$ p(z|x)$，但这在复杂问题里根本算不出来。所以VAE引入一个编码器$q_(phi)(z|x)$，用另一个神经网络去近似它：$ q_(phi)(z|x)=cal(N)(z|mu_(phi)(x),sigma^2_(phi)(x)) $编码器做的事就是：你给它一个销量$x$，它输出一个分布，而不是一个固定的$z$。这个分布由均值$u_(phi)(x)$和方差$sigma^2_(phi)(x)$决定。比如，你输入一个异常高的销量x=3.5，编码器可能输出一个均值在+1.8、方差很小的分布，大致告诉你这个销量最可能来自隐空间里偏「繁荣」状态的那个区域。

== 1.2 证据下界

#h(2em)现在编码器和解码器都有了，但它们都还什么都没学会。我们需要一个统一的目标，让这两个神经网络能通过数据自动“磨合”起来。

#h(2em)这个目标很自然：我们手上有10万条真实销量数据，我们希望整个模型生成的数据越像这些真实数据越好。也就是说，我们希望最大化数据似然：$ 1/N sum_(i=1)^N log p_(theta)(x_i) $而$ p_(theta)(x)=integral p_(theta)(x|z) p(z) d z $然而$ cal(N)(x|mu_(theta)(z),sigma^2) $是个高维的、被复杂非线性函数扭曲的（经过神经网络）、无解析式的数值积分，计算该积分不现实。

#h(2em)编码器成为了救星。我们定义一个新的目标函数，叫做证据下界（ELBO）：$ E L B O(x)=E_(q_(phi)(z|x))[log (p(x,z)) / (q_(phi)(z|x))] $事实上，ELBO是数据似然的一个下界，因为我们可以证明如下：

#at[
  证：由于$ 1=integral q_(phi)(z|x) d z $则$ log p_(theta)(x)=integral q_(phi)(z|x)log p_(theta)(x) d z=E_(q_(phi)(z|x))[log p_(theta)(x)] $而$ p_(theta)(x)=(p(x,z))/(p(z|x)) $则$ E_(q_(phi)(z|x))[log p_(theta)(x)]&=E_(q_(phi)(z|x))[log ((p_(theta)(x,z))/(q_(phi)(z|x)))]+E_(q_(phi)(z|x))[log ((q_(phi)(z|x))/(p_(theta)(z|x)))]\ &=E L B O(x)+K L(q_(phi)(z|x)||p(z|x)) $ 而$ K L(P||Q)=integral P(x)(log P(x) - log Q(x)) d x $令$S={x:P(x)>0}$则$ K L(P||Q)=-E_(x~P)(log Q(x)/P(x)) $由琴生不等式$ -E_(x~P)(log Q(x)/P(x))>=-log E(Q(x)/P(x))=-log integral_S Q(x)d x=0 $则$ K L>=0 $
]

#h(2em)但是这个ELBO仍不实用，因为它涉及$p(x,z)$所以，我们需要继续做些工作：$ E L B O(x)&=E_(q_(phi)(z|x))[log (p(x,z)) / (q_(phi)(z|x))]\ &=E_(q_(phi)(z|x))[log (p(x|z)p(z)) / (q_(phi)(z|x))]\ &=E_(q_(phi)(z|x))[log p(x|z)]+E_(q_(phi)(z|x))[log (p(z))/(q_(phi)(z|x))]\ &=E_(q_(phi)(z|x))[log (p_(theta)(x|z)]-K L(q_(phi)(z|x)||p(z)) $虽然真实的$p(x|z)$我们不知道，但$p_(theta)(x|z)$可以通过解码器近似。它变成了两个可以直接计算、可以优化的项，这就真正实用了。

#h(2em)第一项：重建项——“销量能准确还原吗？”

$ E_(q_(phi) (z|x)) [log p_(theta) (x|z)] $

#h(2em)它做的事很直白：对于给定的真实销量$x$，编码器先猜一个隐变量分布$q_(phi) (z|x)$，从中采样一个$z$，然后让解码器用这个$z$重建销量。因为$p_(theta) (x|z)$是高斯，这个对数似然本质上就是“真实销量”与“解码器预测销量”之间的负均方误差（去掉常数和系数）：

$ E_(q_(phi) (z|x)) [log p_(theta) (x|z)] ∝ -E_(q_(phi) (z|x)) [(x - mu_(theta) (z))^2] $

#h(2em)如果重建得好（比如模型从隐变量推测出今天是一个闷热的夏天，并预测销量在$+3$附近，真实销量恰好也是$+3.2$），这一项就大（ELBO贡献正）；

#h(2em)如果重建失误（模型预测阴冷冬天销量$-2$，实际却是盛夏销量$+3$），这一项就非常负，意味着模型在“惩罚”这种错误编码或错误生成。

#h(2em)这项逼着编码器把销量中的关键信息挤进$z$，也逼着解码器学会从$z$中读取这些信息。

#h(2em)第二项：先验匹配项——“不要偏离标准高斯太远”

$ -K L(q_(phi) (z|x)||p(z)) $

#h(2em)如果我们只保留重建项，编码器会找到一个“作弊”策略：给每一个不同的销量都分配一个彼此相隔极远的隐变量区域，这样解码器可以毫不费力地还原数据，甚至达到完美重建。但这时，隐空间彻底碎裂——没有一个平滑的、可插值的结构，我们也无法从先验$p(z)$中采样生成新数据。

#h(2em)先验匹配项强制每个数据点的编码分布$q_(phi) (z|x)$都尽可能靠近标准高斯$p(z)=cal(N)(0,1)$：

- 这个KL散度项在$q_(phi) (z|x)$也恰好是$cal(N)(0,1)$时为0；
- 均值偏离0越远、方差偏离1越远，惩罚越大。

#h(2em)这两项合在一起，形成一种博弈：

- 重建项要求$q_(phi) (z|x)$含有关于$x$的足够信息；
- KL项要求$q_(phi) (z|x)$不要偏离无信息先验太远。

#h(2em)最终收敛到一种平衡：不同销量的隐变量分布彼此重叠，但整体上自动按销量的相似度排列，形成一个平滑、有意义的隐空间。例如，夏天销量和晚春销量在隐空间里会靠得很近，而深冬销量会被推到另一个方向。之后，只要从先验中随机采样，解码器就能生成各种合理的冰淇淋销量。

#h(2em)然而，ELBO也有其局限性。最大化ELBO并不等价于最大化真实的$log p(x)$，因为ELBO只是一个下界。当编码器容量不足以完全匹配真实后验，或者存在正则化作用时，ELBO得到的最优近似后验$q_(phi)(z|x)$与真实后验$p(z|x)$之间是有差距的。

#h(2em)用冰淇淋的话来说：即使真实的销量完全由温度决定（确定性的线性关系），VAE 为了保证隐空间规整、便于从先验采样，仍然会给编码器“留有余地”，不会把每个销量唯一地映射到一个隐变量点，而是维持一定的模糊性。这种模糊性正是 ELBO 作为下界不得不付出的代价。

== 1.3 VAE 中的优化：如何训练这个模型

#h(2em)前面我们已经把VAE的概率框架搭好了，并且定义了证据下界ELBO作为我们的目标函数。现在是时候面对真正的现实问题：**怎么通过数据来优化那两个神经网络？**

#h(2em)给定训练数据集 $cal(X)={x^((1)),x^((2)),...,x^((L))}$（比如 10 万条冰淇淋销量），VAE 的训练目标是：
$ (phi^*,theta^*)=(a r g m a x)_(phi,theta) sum_(x in cal(X)) E L B O(x) $

#h(2em)其中每个数据点的 ELBO 为：
$ E L B O(x)=E_(q_(phi)(z|x))[log p_(theta)(x|z)]-K L(q_(phi)(z|x)||p(z)) $

#h(2em)直观上，就是我们希望找到一对编码器参数$phi$和解码器参数$theta$，使得所有真实销量的ELBO总和（或平均）最大。

#h(2em)现代神经网络优化依靠梯度下降——先算目标函数关于参数的梯度，再反向传播更新参数。但对VAE来说，这个梯度可没那么好算。

#h(2em)把ELBO写成如下形式（暂时省略KL项，它通常有闭式解，不是难点）：
$ E L B O(x)=E_(q_(phi)(z|x))[f_(theta,phi)(z)] $

#h(2em)其中 $f_(theta,phi)(z)=log p_(theta)(x|z)-log q_(phi)(z|x)$。

#h(2em)问题出在对编码器参数$phi$求梯度的时候。我们想算：
$ nabla_(phi) E_(q_(phi)(z|x))[f(z)] $

#h(2em)但是期望$E_(q_(phi)(z|x))[dot]$依赖于分布$q_(phi)$，而这个分布恰恰是用$phi$参数化的。这意味着我们不能简单地把梯度移进期望里，因为**采样分布本身也在变**。数学上：
$ nabla_(phi) E_(q_(phi)(z|x))[f(z)] != E_(q_(phi)(z|x))[nabla_(phi) f(z)] $

#h(2em)如果强行用蒙特卡洛采样$z^((l)) ~ q_(phi)(z|x)$来近似期望，然后对这些样本算梯度，你会漏掉采样分布随$phi$变化的那部分梯度，导致梯度估计错误。

#h(2em)用冰淇淋的话来说：每次更新编码器参数，它输出的高斯分布（均值、方差）就会变，从而导致你抽取的隐变量$z$的分布也变了。你需要一种方法，能够知道"改变分布参数"会如何影响最终的ELBO，即使采样操作本身是不可导的。

#h(2em)重参数化技巧是 Kingma & Welling 在原始VAE论文中提出的关键创新，它优雅地解决了上述梯度不可算的问题。

#h(2em)核心思想很简单：把随机采样操作从参数依赖中解耦出来。

#h(2em)我们不再直接从$q_(phi)(z|x)$采样，而是：

- 从一个固定的、与$phi$无关的分布中采样一个随机变量$epsilon$（例如标准高斯$epsilon ~ cal(N)(0,bold(I))$）。
- 然后用一个可微的确定性变换$g_(phi)$将$epsilon$变成$z$：
$ z=g_(phi)(epsilon,x) $

#h(2em)并且要求这样得到的$z$恰好服从$q_(phi)(z|x)$。

#h(2em)最常见的例子：如果编码器输出的是高斯分布 $q_(phi)(z|x)=cal(N)(z|mu_(phi)(x),sigma_(phi)^2(x)bold(I))$，那么我们可以令：
$ z=mu_(phi)(x)+sigma_(phi)(x) dot epsilon, quad epsilon ~ cal(N)(0,bold(I)) $

#h(2em)这个操作就是重参数化——现在$z$是一个关于$phi$可微的函数（通过$mu_(phi)$和$sigma_(phi)$），而随机性全部交给了与$phi$无关的$epsilon$。

#h(2em)利用重参数化，ELBO的期望可以重写为：
$ E L B O(x)=E_(epsilon ~ p(epsilon))[f_(theta,phi)(g_(phi)(epsilon,x))] $

#h(2em)这里期望是对固定的$p(epsilon)$取的，与$phi$无关！于是我们可以毫无顾虑地把梯度移进去：
$ nabla_(phi) E L B O(x)=E_(epsilon ~ p(epsilon))[nabla_(phi) f_(theta,phi)(g_(phi)(epsilon,x))] $

#h(2em)现在右边的期望可以简单地用蒙特卡洛估计：
$ approx (1)/(L) sum_(l=1)^L nabla_(phi) f_(theta,phi)(mu_(phi)(x)+sigma_(phi)(x) dot epsilon^((l))), quad epsilon^((l)) ~ cal(N)(0,bold(I)) $

#h(2em)其中对$phi$的梯度可以通过链式法则穿过$mu_(phi)$和$sigma_(phi)$反向传播，因为它们都是神经网络的可微输出。

#h(2em)这就像一个魔术：我们不再对"采样"这一步求导，而是对产生采样均值和方差的网络求导，而噪声$epsilon$像是从外部注入的固定输入，每次都从标准高斯里抓一把即可。

#h(2em)现在我们把上面的想法落实到神经网络的细节上。假设我们的隐变量维度为 $d$。

#h(2em)输入一个冰淇淋销量 $x$（实际可能是向量，比如多个特征；这里简化为一维销量），输出一个高斯分布的参数：
$ (mu_(phi)(x),sigma_(phi)^2(x))=("EncoderNetwork")_(phi)(x) $

#h(2em)然后我们定义近似后验：
$ q_(phi)(z|x)=cal(N)(z|mu_(phi)(x),sigma_(phi)^2(x)bold(I)) $

#h(2em)其中 $mu_(phi)(x) in RR^d$ 是均值向量，$sigma_(phi)^2(x) in RR_+$ 是对角方差（实际中常输出对数方差以保证正值）。

#h(2em)采样时使用重参数化：
$ z=mu_(phi)(x)+sigma_(phi)(x) dot epsilon, quad epsilon ~ cal(N)(0,bold(I)) $

#h(2em)输入隐变量$z$，输出一个关于销量的分布的参数。因为我们假设销量是实数值，通常用一个高斯分布：
$ p_(theta)(x|z)=cal(N)(x|f_(theta)(z),sigma_("dec")^2) $

#h(2em)其中 $f_(theta)(z)$ 是解码器网络（例如 MLP），输出预测的销量均值；$sigma_("dec")$ 是一个固定的超参数，控制生成时的不确定性。也可以让网络输出方差，但固定方差简化了问题。在对数似然中，预测均值 $f_(theta)(z)$ 会直接与真实销量 $x$ 比较。

#h(2em)有了高斯形式的 $q_(phi)(z|x)$ 和高斯先验 $p(z)=cal(N)(0,bold(I))$，KL散度可以写成闭式，无需采样：
$ K L(q_(phi)(z|x)||p(z))=(1)/(2)(sigma_(phi)^2(x) d - d + ||mu_(phi)(x)||^2 - 2d log sigma_(phi)(x)) $

#h(2em)这个公式计算非常快，直接加入损失函数中即可。它惩罚 $mu_(phi)$ 偏离 0 以及 $sigma_(phi)$ 偏离 1 的行为。

#h(2em)对于重建项：
$ E_(q_(phi)(z|x))[log p_(theta)(x|z)] $

#h(2em)我们利用重参数化采一个$z$（实践中通常只采一个样本，即 $L=1$，因为随机梯度下降本身已足够），然后计算对数似然：
$ log p_(theta)(x|z)=-(||x-f_(theta)(z)||^2)/(2sigma_("dec")^2)-(1)/(2)log(2pi sigma_("dec")^2) $

#h(2em)去掉常数，就等价于最小化均方误差$||x-f_(theta)(z)||^2$，其中 $z=mu_(phi)(x)+sigma_(phi)(x) dot epsilon$。

#h(2em)所以整个VAE的损失函数（对单个数据点）可以写成：
$ cal(L)(x;phi,theta)=underbrace((1)/(2sigma_("dec")^2)||x-f_(theta)(mu_(phi)(x)+sigma_(phi)(x) dot epsilon)||^2,"重建误差")+underbrace((1)/(2)(sigma_(phi)^2(x)d - d + ||mu_(phi)(x)||^2 - 2d log sigma_(phi)(x)),"KL 正则") $

#h(2em)（常数项可省略）

#h(2em)遍历训练集中的每个mini-batch，对每个数据点$x$：
- 编码器输出 $mu_(phi)(x),sigma_(phi)(x)$；
- 采样 $epsilon ~ cal(N)(0,bold(I))$，计算 $z=mu_(phi)(x)+sigma_(phi)(x) dot epsilon$；
- 解码器输出$f_(theta)(z)$；
- 计算上述损失，反向传播更新$phi$和$theta$。

#h(2em)因为重参数化使得梯度可以穿过采样步骤，整个网络就是端到端可微的。

#h(2em)训练完毕后，我们扔掉编码器，只保留解码器。生成新销量时：
1. 从先验$cal(N)(0,bold(I))$ 中采样一个$z$；
2. 用解码器计算$f_(theta)(z)$，作为生成销量的均值；
3. 如果需要随机性，可以再添加噪声$sigma_("dec")$ 进行采样。

#h(2em)这样，我们就得到了一个能够从标准高斯噪声生成冰淇淋销量数据的生成模型。

#h(2em)总结一下，该小节的核心就是解决了"VAE 如何通过梯度下降进行训练"这一难题。重参数化技巧把不可微的采样操作变成了可微的变换，让反向传播可以顺畅地进行。同时，具体的编码器和解码器设计（高斯输出，均值由神经网络给出）使得 ELBO 中的所有项都可以高效计算。这些技术共同奠定了 VAE 实用化的基础。
#pagebreak()

= 二、DDPM（降噪扩散概率模型）

#h(2em)在VAE中，我们通过一个隐变量空间来建模数据分布，但它的生成过程是一次性的：从隐变量直接生成数据。从某种意义上说，这对神经网络的要求太高了。很自然地想到，我们可以采用一个逐步的生成过程，通过一系列的迭代步骤，从纯噪声逐渐“净化”成真实数据。也就引出了我们首先要介绍的VDM（变分扩散模型）。

== 2.1 构建模块

#h(2em)让我们讨论变分扩散模型的构建模块。共有三类构建模块：转移块、初始块和最终块。
#image("/assets/images/image.png")
- 转移块：第$t$个转移块包含三个状态$x_(t-1),x_t,x_(t+1)$，到达$x_t$的第一条路径是是$x_(t-1)->x_t$的前向转移，相关的转移分布是$p(x_t|x_(t-1))$，这类似于VAE中的编码器，同样地，该分布不可访问，我们可以用高斯分布来近似，近似记作$q_(phi) (x_t|x_(t-1))$。第二条路径是$x_(t+1)->x_t$的反向转移，相关的转移分布是$p(x_t|x_(t+1))$，这类似于VAE中的解码器，同样地，该分布不可访问，我们可以用高斯分布来近似，近似记作$p_(theta) (x_t|x_(t+1))$。

- 初始块：初始块包含一个状态$x_0$，它是我们想要建模的数据分布的样本。我们只需考虑反向转移分布$p_(theta) (x_0|x_1)$。

- 最终块：最终块包含一个状态$x_T$，它是一个纯噪声分布的样本。我们只需考虑前向转移分布$q_(phi) (x_T|x_(T-1))$。

#h(2em)我们先来定义转移分布$ q_(phi) (x_t|x_(t-1))=cal(N)(x_t|sqrt(alpha_t)x_(t-1),(1-alpha_t)I) $后续我们会解释这样定义的合理性，现在我们先给出结论：该迭代过程会在平衡态（$x->oo$）时给出白高斯。

#h(2em)假设$q_(phi) (x_t|x_(t-1))=cal(N)(x_t|a x_(t-1),b^2 I) $，如果我们希望在迭代足够多次后得到$cal(N)(0,I)$，那么我们需要满足以下条件：$cases(a=sqrt(alpha),b=sqrt(1-alpha))$。这是因为对于该分布，等价的采样为$x_t=a x_(t-1)+b epsilon_(t-1)$，其中$epsilon_(t-1) ~ cal(N)(0,I)$。递归得$ x_t&=a x_(t-1)+b epsilon_(t-1)\ &=dots\ &=a^t x_0+underbrace(b(epsilon_(t-1)+a epsilon_(t-2)+...+a^(t-1) epsilon_(0)),w_t ) $则$E(w_t)=0$，那么协方差矩阵$ C o v(w_t)=E(w_t w_t^T)=b^2 dot frac(1-a^(2t),1-a^2)I $故$ lim_(t->oo) C o v(w_t)=(b^2)/(1-a^2) I $若希望其为$I$，只需$a^2+b^2=1$得证。

#h(2em)更进一步地，$ q_(phi) (x_t|x_0)=cal(N)(x_t|sqrt(overline(alpha_t))x_0,(1-overline(alpha_t))I),overline(alpha_t)=product_(s=1)^t alpha_s $

#at[
  证：递归展开
  $ x_t &= sqrt(alpha_t) x_(t-1) + sqrt(1-alpha_t) epsilon_(t-1) \
  &= sqrt(alpha_t alpha_(t-1)) x_(t-2) + underbrace(sqrt(alpha_t) sqrt(1-alpha_(t-1)) epsilon_(t-2), w_1) + underbrace(sqrt(1-alpha_t) epsilon_(t-1), w_2) $
  两个高斯之和仍为高斯，其协方差为
  $ E(w_1 w_1^T) = ( (sqrt(alpha_t) sqrt(1-alpha_(t-1)))^2 + (sqrt(1-alpha_t))^2 ) bold(I) = (1 - alpha_t alpha_(t-1)) bold(I) $
  继续递归得
  $ x_t = sqrt(product_(i=1)^t alpha_i) x_0 + sqrt(1 - product_(i=1)^t alpha_i) epsilon_0 $
  定义 $overline(alpha_t) = product_(i=1)^t alpha_i$，即得
  $ q_(phi) (x_t|x_0) = cal(N)(x_t | sqrt(overline(alpha_t)) x_0, (1-overline(alpha_t)) bold(I)) $
]

#h(2em)这个新分布 $q_(phi) (x_t|x_0)$ 的用处在于它给出了单步前向扩散，而非链式 $x_0 -> x_1 -> dots -> x_T$。在前向扩散的每一步，由于我们已知 $x_0$ 并假设所有后续转移都是高斯的，我们可以直接知道任意 $t$ 下的 $x_t$。

#h(2em)此外，对于高斯混合模型 $x_0 ~ p_0 (x) = sum_(k=1)^K pi_k cal(N)(x | mu_k, sigma_k^2 bold(I))$，在时间 $t$ 的分布为
$ x_t ~ p_t (x) = sum_(k=1)^K pi_k cal(N)(x | sqrt(overline(alpha_t)) mu_k, (1-overline(alpha_t)) bold(I) + overline(alpha_t) sigma_k^2 bold(I)) $

#h(2em)到目前为止我们仅仅在讨论前向过程。在扩散模型中，前向过程被选择为能够以闭合形式表达。更有趣的部分是反向过程——它通过一系列去噪操作实现，每个去噪步骤都应与其在前向过程中的对应步骤耦合。$q_(phi) (x_t|x_0)$ 只是提供了一种更方便实现前向过程的方式。

== 2.2 证据下界

#h(2em)现在我们理解了变分扩散模型的结构，可以写出 ELBO 并因此训练模型。

#at[
  #the[定理 2.3（变分扩散模型的 ELBO）] 变分扩散模型的 ELBO 为
  $ E L B O_(phi,theta) (x) &= E_(q_(phi) (x_1|x_0)) [log underbrace(p_(theta) (x_0|x_1), "初始块")] \
  &quad - E_(q_(phi) (x_(T-1)|x_0)) [underbrace(D_(K L) (q_(phi) (x_T|x_(T-1)) || p(x_T)), "最终块")] \
  &quad - sum_(t=1)^(T-1) E_(q_(phi) (x_(t-1), x_(t+1)|x_0)) [underbrace(D_(K L) (q_(phi) (x_t|x_(t-1)) || p_(theta) (x_t|x_(t+1))), "过渡块")] $
  其中 $x_0 = x$，$x_T ~ cal(N)(0, bold(I))$。
]

#h(2em)我们逐项理解。

#h(2em)第一项是 #strong[重建（初始块）]：$E_(q_(phi) (x_1|x_0)) [log p_(theta) (x_0|x_1)]$。期望内部是对数似然 $log p_(theta) (x_0|x_1)$，它度量了神经网络从隐变量 $x_1$ 恢复 $x_0$ 的能力。期望是关于从 $q_(phi) (x_1|x_0)$ 抽取的样本取的——因为 $x_1$ 不是凭空而来，而是由前向转移 $q_(phi) (x_1|x_0)$ 创建的。

#h(2em)第二项是 #strong[先验匹配（最终块）]：$-E_(q_(phi) (x_(T-1)|x_0)) [D_(K L) (q_(phi) (x_T|x_(T-1)) || p(x_T))]$。我们使用 KL 散度度量 $q_(phi) (x_T|x_(T-1))$ 与 $p(x_T)$ 的差异。由于我们假设 $p(x_T) = cal(N)(0, bold(I))$，我们希望 $q_(phi) (x_T|x_(T-1))$ 尽可能接近白高斯。

#h(2em)第三项是 #strong[一致性（过渡块）]：$-sum_(t=1)^(T-1) E_(q_(phi) (x_(t-1), x_(t+1)|x_0)) [D_(K L) (q_(phi) (x_t|x_(t-1)) || p_(theta) (x_t|x_(t+1)))]$。前向转移由 $q_(phi) (x_t|x_(t-1))$ 决定，反向转移由 $p_(theta) (x_t|x_(t+1))$ 决定。一致性项使用 KL 散度度量二者的偏差。

#at[
  证（定理 2.3）：记 $x_(0:T) = {x_0, dots, x_T}$。先验 $p(x) = p(x_0)$。则
  $ log p(x) = log p(x_0) = log integral p(x_(0:T)) d x_(1:T) $
  乘以并除以 $q_(phi) (x_(1:T)|x_0)$：
  $ = log integral (p(x_(0:T)) q_(phi) (x_(1:T)|x_0)) / (q_(phi) (x_(1:T)|x_0)) d x_(1:T) = log E_(q_(phi) (x_(1:T)|x_0)) [p(x_(0:T)) / q_(phi) (x_(1:T)|x_0)] $
  由琴生不等式（$log$ 为凹函数）：
  $ >= E_(q_(phi) (x_(1:T)|x_0)) [log p(x_(0:T)) / q_(phi) (x_(1:T)|x_0)] $
  展开 $p(x_(0:T)) = p(x_T) product_(t=1)^T p(x_(t-1)|x_t)$ 和 $q_(phi) (x_(1:T)|x_0) = product_(t=1)^T q_(phi) (x_t|x_(t-1))$，代入并利用条件独立性即得定理中的 ELBO。
]

#h(2em)然而，在上述 ELBO 中，我们需要从联合分布 $q_(phi) (x_(t-1), x_(t+1)|x_0)$ 中抽取样本对 $(x_(t-1), x_(t+1))$。我们不知道这个联合分布是什么，而且使用未来样本 $x_(t+1)$ 来抽取当前样本 $x_t$ 也很奇怪。

#h(2em)这里有一个简单的技巧——贝叶斯定理：
$ q(x_t|x_(t-1)) = (q(x_(t-1)|x_t) q(x_t)) / q(x_(t-1)) quad 以 x_0 为 条 件 => quad q(x_t|x_(t-1), x_0) = (q(x_(t-1)|x_t, x_0) q(x_t|x_0)) / q(x_(t-1)|x_0) $

#h(2em)通过增加条件变量 $x_0$，我们将 $q(x_t|x_(t-1), x_0)$ 转换为 $q(x_(t-1)|x_t, x_0)$。方向 $q(x_(t-1)|x_t, x_0)$ 现在与 $p_(theta) (x_(t-1)|x_t)$ 平行。因此，我们可以重写一致性项为 $q_(phi) (x_(t-1)|x_t, x_0)$ 与 $p_(theta) (x_(t-1)|x_t)$ 之间的 KL 散度。

#at[
  #the[定理 2.4（重写后的 ELBO）] 令 $x = x_0$，$x_T ~ cal(N)(0, bold(I))$。定理 2.3 的 ELBO 可等价写为
  $ E L B O_(phi,theta) (x) &= E_(q_(phi) (x_1|x_0)) [log p_(theta) (x_0|x_1)] \
  &quad - underbrace(D_(K L) (q_(phi) (x_T|x_0) || p(x_T)), "新的先验匹配") \
  &quad - sum_(t=2)^T underbrace(E_(q_(phi) (x_t|x_0)) [D_(K L) (q_(phi) (x_(t-1)|x_t, x_0) || p_(theta) (x_(t-1)|x_t))], "新的一致性") $
]

#h(2em)变化有三：
- #strong[重建]：不变，仍在最大化对数似然。
- #strong[先验匹配]：简化为 $q_(phi) (x_T|x_0)$ 与 $p(x_T)$ 的 KL 散度，不再需要对 $x_(T-1)$ 取期望。
- #strong[一致性]：索引从 $t=2$ 到 $T$（此前是 $t=1$ 到 $T-1$），匹配对象变为 $q_(phi) (x_(t-1)|x_t, x_0)$ 与 $p_(theta) (x_(t-1)|x_t)$。

== 2.3 反向过程的分布

#h(2em)现在我们讨论新 ELBO 的核心组成部分 $q_(phi) (x_(t-1)|x_t, x_0)$。关键结论：它仍然是高斯的，完全由均值和协方差表征。

#at[
  #the[定理 2.5] 分布 $q_(phi) (x_(t-1)|x_t, x_0)$ 的形式为
  $ q_(phi) (x_(t-1)|x_t, x_0) = cal(N)(x_(t-1) | mu_q (x_t, x_0), Sigma_q (t)) $
  $ mu_q (x_t, x_0) = ((1-overline(alpha_(t-1))) sqrt(alpha_t)) / (1-overline(alpha_t)) x_t + ((1-alpha_t) sqrt(overline(alpha_(t-1)))) / (1-overline(alpha_t)) x_0 $
  $ Sigma_q (t) = ((1-alpha_t)(1-overline(alpha_(t-1)))) / (1-overline(alpha_t)) bold(I) =: sigma_q^2 (t) bold(I) $
  其中 $overline(alpha_t) = product_(i=1)^t alpha_i$。
]

#h(2em)均值 $mu_q (x_t, x_0)$ 是 $x_t$ 和 $x_0$ 的线性组合——几何上它位于连接 $x_t$ 和 $x_0$ 的直线上。

#at[
  证（定理 2.5）：利用贝叶斯定理
  $ q(x_(t-1)|x_t, x_0) = (cal(N)(x_t|sqrt(alpha_t) x_(t-1), (1-alpha_t) bold(I)) cal(N)(x_(t-1)|sqrt(overline(alpha_(t-1))) x_0, (1-overline(alpha_(t-1))) bold(I))) / (cal(N)(x_t|sqrt(overline(alpha_t)) x_0, (1-overline(alpha_t)) bold(I))) $
  为简单起见将向量视为标量。上述高斯乘积正比于
  $ exp {(x_t - sqrt(alpha_t) x_(t-1))^2 / (2(1-alpha_t)) + (x_(t-1) - sqrt(overline(alpha_(t-1))) x_0)^2 / (2(1-overline(alpha_(t-1)))) - (x_t - sqrt(overline(alpha_t)) x_0)^2 / (2(1-overline(alpha_t)))} $
  考虑二次函数 $f(y) = (x - sqrt(a) y)^2/(2(1-a)) + (y - sqrt(b) z)^2/(2(1-b)) - (x - sqrt(c) z)^2/(2(1-c))$。求导：
  $ f'(y) = (1-a b)/((1-a)(1-b)) y - (sqrt(a)/(1-a) x + sqrt(b)/(1-b) z) $
  令 $f'(y)=0$ 得 $y = ((1-b) sqrt(a))/(1-a b) x + ((1-a) sqrt(b))/(1-a b) z$。
  注意到 $a b = alpha_t overline(alpha_(t-1)) = overline(alpha_t)$，代入即得 $mu_q$ 的表达式。
  对于方差，检查曲率 $f''(y) = (1-a b)/((1-a)(1-b)) = (1-overline(alpha_t))/((1-alpha_t)(1-overline(alpha_(t-1))))$，取倒数即得 $Sigma_q (t)$。
]

#h(2em)随着 $t$ 从 $T$ 减小到 $1$，$x_t$ 的系数缩小而 $x_0$ 的系数增大。方差 $sigma_q^2 (t)$ 在 $t=T$ 时较大（使 $x_t$ 更接近白高斯），在 $t=1$ 时趋于零（最终 $x_0$ 应为无噪声的干净图像）。

#h(2em)#strong[构造 $p_(theta) (x_(t-1)|x_t)$。] 上式揭示了一个令人振奋的的事实：$q_(phi) (x_(t-1)|x_t, x_0)$ 完全由 $x_t$ 和 $x_0$ 表征——不需要神经网络来估计均值和方差！一旦超参数 $alpha_t$ 被定义，分布就确定了。因此一致性项中，唯一需要"学习"的部分是 $p_(theta) (x_(t-1)|x_t)$。

#h(2em)自然的想法是将 $p_(theta) (x_(t-1)|x_t)$ 也故意定义为同方差的高斯分布：
$ p_(theta) (x_(t-1)|x_t) = cal(N)(x_(t-1) | mu_(theta) (x_t), sigma_q^2 (t) bold(I)) $
其中 $mu_(theta) (x_t)$ 由神经网络估计，方差 $sigma_q^2 (t) bold(I)$ 设为已知常数。利用两个高斯分布之间 KL 散度的公式：
$ D_(K L) (q_(phi) (x_(t-1)|x_t, x_0) || p_(theta) (x_(t-1)|x_t)) = 1/(2 sigma_q^2 (t)) || mu_q (x_t, x_0) - mu_(theta) (x_t) ||^2 $

#h(2em)因此 ELBO 变为
$ E L B O_(theta) (x) = E_(q(x_1|x_0)) [log p_(theta) (x_0|x_1)] - sum_(t=2)^T E_(q(x_t|x_0)) [1/(2 sigma_q^2 (t)) || mu_q (x_t, x_0) - mu_(theta) (x_t) ||^2] $
这里省略了 $D_(K L) (q(x_T|x_0) || p(x_T))$ 项，因为它不依赖于 $theta$。

== 2.4 训练和推理

#h(2em)我们需要找到网络 $mu_(theta)$ 最小化 $||mu_q (x_t, x_0) - mu_(theta) (x_t)||^2$。回忆
$ mu_q (x_t, x_0) = ((1-overline(alpha_(t-1))) sqrt(alpha_t)) / (1-overline(alpha_t)) x_t + ((1-alpha_t) sqrt(overline(alpha_(t-1)))) / (1-overline(alpha_t)) x_0 $

#h(2em)我们将网络参数化为让网络预测干净图像 $hat(x)_(theta) (x_t)$，然后与已知系数组合：
$ mu_(theta) (x_t) = underbrace(((1-overline(alpha_(t-1))) sqrt(alpha_t)) / (1-overline(alpha_t)) x_t, "已知") + underbrace(((1-alpha_t) sqrt(overline(alpha_(t-1)))) / (1-overline(alpha_t)) hat(x)_(theta) (x_t), "网络") $

#h(2em)代入得
$ 1/(2 sigma_q^2 (t)) || mu_q (x_t, x_0) - mu_(theta) (x_t) ||^2 = 1/(2 sigma_q^2 (t)) ((1-alpha_t)^2 overline(alpha_(t-1))) / (1-overline(alpha_t))^2 || hat(x)_(theta) (x_t) - x_0 ||^2 $

#h(2em)将第一项 $E_(q(x_1|x_0)) [log p_(theta) (x_0|x_1)]$ 也吸收进求和中，得到统一的 ELBO：

#at[
  #the[定理 2.7（DDPM 的 ELBO）] 去噪扩散概率模型的 ELBO 为
  $ E L B O_(theta) (x) = -sum_(t=1)^T 1/(2 sigma_q^2 (t)) ((1-alpha_t)^2 overline(alpha_(t-1))) / (1-overline(alpha_t))^2 E_(q(x_t|x_0)) [|| hat(x)_(theta) (x_t) - x_0 ||^2] $
]

#h(2em)忽略常数和期望，对于特定的 $x_t$，关注的重点是
$ (a r g m i n)_(theta) || hat(x)_(theta) (x_t) - x_0 ||^2 $
这不过是一个去噪问题——我们需要找到一个网络 $hat(x)_(theta)$ 使得去噪后的图像接近真实图像 $x_0$。使其不同于典型去噪器的是：
- $E_(q(x_t|x_0))$：噪声图像并非任意随机，而是精心选择的 $x_t = sqrt(overline(alpha_t)) x_0 + sqrt(1-overline(alpha_t)) epsilon_t$，其中 $epsilon_t ~ cal(N)(0, bold(I))$。
- 权重 $1/(2 sigma_q^2 (t)) ((1-alpha_t)^2 overline(alpha_(t-1))) / (1-overline(alpha_t))^2$：并非对所有步骤等同加权，而是存在一个调度器控制每个去噪损失的相对重点。

#h(2em)由于对分布$q(x_t|x_0)$中的每个$x$采样无法做到，因此同样地，使用蒙特卡洛近似期望，优化问题写为
$ (a r g m i n)_(theta) sum_(x_0 in cal(X)) sum_(t=1)^T 1/M sum_(m=1)^M 1/(2 sigma_q^2 (t)) ((1-alpha_t)^2 overline(alpha_(t-1))) / (1-overline(alpha_t))^2 || hat(x)_(theta) (sqrt(overline(alpha_t)) x_0 + sqrt(1-overline(alpha_t)) epsilon_t^((m))) - x_0 ||^2 $
其中 $epsilon_t^((m)) ~ cal(N)(0, bold(I))$。所得模型被称为 #strong[去噪扩散概率模型（DDPM）]。

#h(2em)#strong[DDPM 的训练算法。] 对于训练数据集中的每张图像 $x_0$，重复以下步骤直到收敛：
- 选取随机时间戳 $t ~ "Uniform"[1, T]$。
- 抽取样本 $x_t^((m)) = sqrt(overline(alpha_t)) x_0 + sqrt(1-overline(alpha_t)) epsilon_t^((m))$，$epsilon_t^((m)) ~ cal(N)(0, bold(I))$。
- 对梯度 $nabla_(theta) {1/M sum_(m=1)^M || hat(x)_(theta) (x_t^((m))) - x_0 ||^2}$ 采取梯度下降步。

#h(2em)这里我们正在为所有噪声条件训练一个去噪网络 $hat(x)_(theta)$。

#h(2em)#strong[DDPM 的推理——反向扩散。] 一旦去噪器 $hat(x)_(theta)$ 被训练好，推理是从分布 $p_(theta) (x_(t-1)|x_t)$ 中沿状态序列 $x_T, x_(T-1), dots, x_1$ 采样图像。通过重参数化：
$ x_(t-1) = mu_(theta) (x_t) + sigma_q (t) epsilon, quad epsilon ~ cal(N)(0, bold(I)) $
$ = ((1-overline(alpha_(t-1))) sqrt(alpha_t)) / (1-overline(alpha_t)) x_t + ((1-alpha_t) sqrt(overline(alpha_(t-1)))) / (1-overline(alpha_t)) hat(x)_(theta) (x_t) + sigma_q (t) epsilon $

#h(2em)#strong[DDPM 的推理算法：]
- 给定白噪声向量 $x_T ~ cal(N)(0, bold(I))$。
- 对 $t = T, T-1, dots, 1$ 重复：
  - 使用训练好的去噪器计算 $hat(x)_(theta) (x_t)$。
  - 更新 $x_(t-1) = ((1-overline(alpha_(t-1))) sqrt(alpha_t)) / (1-overline(alpha_t)) x_t + ((1-alpha_t) sqrt(overline(alpha_(t-1)))) / (1-overline(alpha_t)) hat(x)_(theta) (x_t) + sigma_q (t) epsilon$，$epsilon ~ cal(N)(0, bold(I))$。

== 2.5 预测噪声

#h(2em)通过 $x_t = sqrt(overline(alpha_t)) x_0 + sqrt(1-overline(alpha_t)) epsilon$ 重新参数化，我们可以训练网络预测噪声 $epsilon_(theta) (x_t, t)$ 而非干净图像 $hat(x)_(theta) (x_t)$。

#h(2em)#strong[训练。] 损失函数转化为
$ cal(L) = E_(t, x_0, epsilon) [|| epsilon - epsilon_(theta) (x_t, t) ||^2] $
训练算法：采样 $x_0$，采样 $t$，采样噪声 $epsilon$，构造 $x_t = sqrt(overline(alpha_t)) x_0 + sqrt(1-overline(alpha_t)) epsilon$，预测噪声，计算 MSE。

#h(2em)#strong[推理。] 均值函数用 $epsilon_(theta)$ 表示：
$ mu_(theta) (x_t) = 1/(sqrt(alpha_t)) (x_t - (1-alpha_t)/(sqrt(1-overline(alpha_t))) epsilon_(theta) (x_t, t)) $
因此推理步骤变为
$ x_(t-1) = 1/(sqrt(alpha_t)) (x_t - (1-alpha_t)/(sqrt(1-overline(alpha_t))) epsilon_(theta) (x_t, t)) + sigma_q (t) bold(z), quad bold(z) ~ cal(N)(0, bold(I)) $

== 2.6 去噪扩散隐式模型（DDIM）

#h(2em)#strong[从 DDPM 到 DDIM。] DDPM 最普遍的缺点之一是需要大量迭代才能生成合理好看的图像。如 Song 等人所述，DDPM 在标准 GPU 上生成 50k 张 $256 times 256$ 图像需要超过 1000 小时。因此有必要减少迭代次数，DDIM 正是为此而发明。

#h(2em)回忆原始 DDPM 转移概率 $q(x_t|x_(t-1)) = cal(N)(x_t|sqrt(alpha_t) x_(t-1), (1-alpha_t) bold(I))$，以及 $q(x_t|x_0) = cal(N)(x_t|sqrt(overline(alpha_t)) x_0, (1-overline(alpha_t)) bold(I))$。转移概率遵循马尔可夫链——$x_t$ 仅依赖于 $x_(t-1)$。马尔可夫结构的优点是无记忆，但缺点是需要很多步才能收敛。DDIM 通过从马尔可夫结构转向非马尔可夫结构克服了这个问题。

#h(2em)#strong[DDIM 中的概率分布。] 跟随 Song 等人的参数选择，将 $alpha_t$ 替换为比值 $alpha_t/alpha_(t-1)$：
$ q(x_t|x_(t-1)) = cal(N)(x_t | sqrt(alpha_t/alpha_(t-1)) x_(t-1), (1 - alpha_t/alpha_(t-1)) bold(I)) $
通过这种构造，可以保持 $q(x_t|x_0)$ 的原始形式。DDIM 的核心思想是定义一个新的前向转移分布族，该分布族是非马尔可夫的，但保持相同的 $q(x_t|x_0)$。引入超参数 $sigma_t$：

#at[
  #the[定理 2.9（DDIM 转移分布）] 在 DDIM 中，转移分布定义为
  $ q(x_(t-1)|x_t, x_0) = cal(N)(sqrt(alpha_(t-1)) x_0 + sqrt(1-alpha_(t-1)-sigma_t^2) ((x_t - sqrt(alpha_t) x_0) / sqrt(1-alpha_t)), sigma_t^2 bold(I)) $
  若 $q(x_t|x_0) = cal(N)(sqrt(alpha_t) x_0, (1-alpha_t) bold(I))$，则 $q(x_(t-1)|x_0) = cal(N)(sqrt(alpha_(t-1)) x_0, (1-alpha_(t-1)) bold(I))$。
]

#h(2em)#strong[DDIM 的推理。] 利用网络预测噪声 $epsilon_(theta)^((t)) (x_t)$，估计干净图像
$ bold(f)_(theta)^((t)) (x_t) = 1/(sqrt(alpha_t)) (x_t - sqrt(1-alpha_t) epsilon_(theta)^((t)) (x_t)) $
然后采样步骤为
$ x_(t-1) = sqrt(alpha_(t-1)) underbrace(((x_t - sqrt(1-alpha_t) epsilon_(theta)^((t)) (x_t)) / sqrt(alpha_t)), "预测的" x_0) + underbrace(sqrt(1-alpha_(t-1)-sigma_t^2) dot epsilon_(theta)^((t)) (x_t), "指向" x_t "的方向") + underbrace(sigma_t epsilon_t, ~ cal(N)(0, bold(I))) $

#h(2em)将此方程与 DDPM 的方程比较：
$ ("DDPM") quad x_(t-1) = 1/(sqrt(alpha_t)) (x_t - (1-alpha_t)/(sqrt(1-overline(alpha_t))) epsilon_(theta)^((t)) (x_t)) + sigma_q (t) epsilon_t $

#h(2em)DDPM 与 DDIM 之间的主要区别很微妙。虽然它们都使用 $x_t$ 和 $epsilon^((t)) (x_t)$ 进行更新，但具体的更新公式导致了不同的收敛速度。事实上，在后来将 DDIM 和 DDPM 与随机微分方程联系起来的微分方程文献中，观察到 DDIM 在求解微分方程时采用了一些特殊的加速一阶数值格式。

