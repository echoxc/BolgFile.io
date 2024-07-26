---
title: Memo - 反应堆物理备忘录
date: 2024-05-06 19:25:34
mathjax: true
top: 70
categories: # 分类
- Physics
tags: # 标签
- 反应堆物理
---

# 基本知识补充
<!--more-->
## 核燃料

* **UOX燃料** ：UOX代表低浓缩铀氧化物（Uranium Oxide），它是一种常见的核燃料形式，UOX燃料通常使用天然铀或低浓缩铀（$^{235}{\rm U}$含量在3-5%之间）制成。
* **MOX燃料**：MOX代表混合氧化物（Mixed Oxide），它混合了铀和钚，是由${\rm UO}_{2}$和${\rm PuO}_{2}$构成的氧化铀钚燃料。MOX燃料通常由浓缩铀和从核废料中回收的钚混合而成。

UOX和MOX燃料都是核反应堆的燃料形式，但MOX燃料相比UOX燃料含有更高比例的$^{235}{\rm U}$以及$^{239}{\rm Pu}$

# 基本概念常识补充

# 常见缩写
## 几种反应堆型
* **LMFBR**: Liquid Metal Fast Breeder Reactors 液态金属快增殖反应堆
* **BWR**: Boiling Water Reactor 沸水反应堆
* **PWR**: Pressurized Water Reactor 压水反应堆
* **CANDU**: 加拿大重水铀反应堆

## 几个实验室缩写
* **ORNL**：Oak Ridge National Laboratory 美国橡树岭国家实验室
* **ANL**：Argonne National Laboratory 美国阿贡国家实验室
* **JAEA**：Japaen Atomic Engery Agency 日本原子能机构
* **OECD/NEA**：Organization for Economic Co-operation and Development Nuclear Energy Agency 国际经济合作与发展组织核能机构
* **BNL**：Brookhaven National Laboratory 鲁克海文国家实验室
* **LANL**：Los Alamos National Laboratory 洛斯阿拉莫斯国家实验室

## 几个核数据库
* **ENDF**

# 专业名词解释

## EOI

"EOI" 代表 "End Of Irradiation"，即“辐照结束”，用来指代燃料组件或单个燃料棒在反应堆中辐照的结束点。在EOI时刻，燃料已经达到了其设计寿命的终点，此时会进行停堆，并开始后续的冷却、运输和处理过程。

在核燃料的背景下，EOI是重要的时间节点: 

* 它标志着燃料从反应堆中移出，开始冷却期。

* 它影响燃料的放射性和热输出，因为随着时间的推移，短寿命的放射性核素会衰变。

* 它决定了燃料中裂变产物和转换材料的库存量，这些材料的组成是评估燃料特性和后续处理选项的关键。

# 核反应堆计算程序

## ORIGEN (ORNL Isotope Generation and Depletion code)

### ORIGEN-2 Version

**ORIGEN-2**是一个著名的零维燃耗计算程序，模拟核燃料在反应堆中因中子辐照而发生的燃耗过程，可以预测燃料在辐照过程中的同位素组成变化。

它不考虑空间维度（即没有考虑燃料棒或燃料组件内部的几何结构和空间分布），而是集中在燃料的化学和核物理特性上，特别是关注燃料中同位素的组成随时间的变化。由于零维模型没有考虑空间异质性，因此它们不能提供空间上燃耗分布的细节。对于需要考虑空间效应的复杂分析，可能需要使用更高维度的燃耗模型。

### ORIGEN-APR Version

**ORIGEN-APR**（ORIGEN-Advanced Physics and Reactor）是从ORIGEN系列代码发展而来，这些代码最初被设计用于计算核材料中同位素的组成和特性。

ORIGEN-APR特别关注于提供更高级的物理模型和更精确的燃耗计算，包括以下特点：

* 更精确的物理模型：ORIGEN-APR包含了更精确的核数据和更详细的物理过程描述，如中子俘获、裂变、衰变等。
* 多维燃耗模拟：与简单的零维模型不同，ORIGEN-APR能够进行**一维或多维燃耗模拟**，这意味着它可以模拟燃料棒内部或整个反应堆芯的燃耗分布。
* 先进的反应堆物理：该程序可以模拟不同类型的反应堆，包括压水堆（PWR）、沸水堆（BWR）、重水堆（CANDU）、快中子堆（如LMFBR）等，并考虑了各自的中子能谱特性。
* 考虑冷却时间：ORIGEN-APR能够考虑燃料辐照结束后的冷却时间对同位素组成的影响，这对于核材料的管理和处置非常重要。

# 常见反应堆

## VVER

参见[VVER反应堆简介](https://www.bilibili.com/read/cv25679314/)

水-水高能反应堆（Water-water energetic reactor），简称WWER或VVER，为苏联于上世纪70年代研发的压水反应堆，与美国的PWR相比，特点为采用卧式蒸汽发生器、六边形燃料组件、无底部穿孔压力容器和高容量稳压器。

## 组件示意图



## 反应堆参数

* **参考文献**  
1. Bilodid, Y., Fridman, E., & Lötsch, T. X2 VVER-1000 benchmark revision: Fresh HZP core state and the reference Monte Carlo solution. Annals of Nuclear Energy, **144**, 107558 (2020).
2. Hossain, M. I., Mollah, A. S., Akter, Y., & Fardin, M. Z. Neutronic calculations for the VVER-1000 MOX core computational benchmark using the OpenMC code. Nuclear Energy and Technology, **9**, 215 (2023).
3. Khuwaileh, B. A., Al-Shabi, M., & Assad, M. E. H. Artificial Neural Network based Particle Swarm Optimization solution approach for the inverse depletion of used nuclear fuel. Annals of Nuclear Energy, **157**, 108256 (2021).

