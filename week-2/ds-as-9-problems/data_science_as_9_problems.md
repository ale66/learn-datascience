---
lang: en
author: DSTA
title: "Data Science as 9 problems"
date: 
---


## A gentle-yet-focussed introduction

![[Ch. 2](./provost-ch2.pdf)](./imgs/provost-cover.jpg)

-----

1. (was 2) Regression/value estimation

__Instance:__

* a collection (dataset) of numerical $<\mathbf{x}, y>$ datapoints

* a regressor (independent) value $\mathbf{x}$

. . .

__Solution:__  a regressand (dependent) value $y$

that complements $\mathbf{x}$

__Measure:__ error over the collection

-----

2. (was 1) Classification and class probability

__Instance:__

* a collection (dataset) of datapoints from $\mathbf{X}$

* a classification system $C = \{c_1, c_2, \dots c_k\}$

. . .

__Solution:__  classification function $\gamma: \mathbf{X} \rightarrow C$

__Measure:__ misclassification

. . .

[PF] "classification predicts whether something will happen, whereas regr. predicts how much something will happen."

-----

![Type I and II errors](./imgs/false_positives_and_negatives.jpg)

-----

3. Similarity

Identify similar individuals based on data known about them.

__Instance:__

* a collection (dataset) of datapoints from $\mathbf{X}$, e.g., $\mathbb{R}^n$

* (distance functions for some of the dimensions)

. . .

__Solution:__  similarity function $\sigma: \mathbf{X} \rightarrow \mathbb{R}$

[__Measure:__ error]

-----

![[Ch. 2](./provost-ch2.pdf)](./imgs/similarity.png)

<!-- http://dingcvnote.blogspot.com/2018/06/matlab-comparing-of-detect-feature.html -->

-----

4. Clustering (segmentation)

group individuals in a population together by their similarity (but not driven by any specific purpose)

__Instance:__

* a collection (dataset) $\mathbf{D}$ of datapoints from $\mathbf{X}$, e.g., $\mathbb{R}^n$

* a relational structure on $\mathbf{X}$ (a graph)

* a small integer $k$

. . .

__Solution:__  a partition of $\mathbf{D}$ into $\mathcal{C_1}, \dots \mathcal{C_k}$

__Measure:__ network modularity Q: proportion of the relational structure that _respects_ the clusters.

-----

Detection version: $k$ is part of the output.

See an [example research work  (from yours truly)](
https://www.sciencedirect.com/science/article/pii/S0022000013000767)

-----

5. Co-occurence (frequent itemset mining)

similarity of objects based on their appearing together in transactions.

__Instance:__

* a collection (dataset) $\mathbf{T}$ of itemsets (subsets of  $\mathbf{X}$) or sequences

* a theshold $\tau$

. . .

__Solution:__  All _frequent patterns:_ subsets that appear in $\mathbf{T}$ above $\tau$

. . .

Detection version: $\tau$ is part of the output.

Market-basket analysis, (some) recommendation systems

-----

6. Profiling (behaviour description)

__Instance:__

* a user description $\mathbf{u}$ drawn from a $\mathbf{D}$ collection

* a stimulus $a\in \mathbf{A}$

* a set of possible responses $\mathbf{R}$

. . .

__Solution:__  a functional reaction of __u__ to __a__, i.e., $\rho: \mathbf{U} \times \mathbf{A} \rightarrow \mathbf{R}$

. . .

Application: anomaly/fraud detection.

Example research work on [Social media profiling](https://ieeexplore.ieee.org/abstract/document/6994286)

-----

7. Link prediction

__Instance:__  a dynamical [graph (network) $\mathbf{G}$](https://en.wikipedia.org/wiki/Graph_(discrete_mathematics)) , i.e., a sequence

$<V, E>$,

$<V, E^\prime=E+\{(u,v)\}>$,

$<V, E^{\prime\prime}=E^\prime+\{(r,s)\}>\dots$

![](./imgs/network.png)

-----

__Question:__  what is the next link to be created?

What YouTube video will you watch next?

Alternatives: predict the __strength__ of the new link; link deletion.

-----

8. Data reduction

__Instance:__

* a collection (dataset) $\mathbf{D}$ of datapoints from $\mathbf{X}$, e.g., $\mathbb{R}^m$

* [a distinct independent variable $x_i$]

. . .

__Solution:__  a projection of $\mathbf{D}$ onto $\mathbb{R}^n$, $n < m$

__Measure:__ error in the estimation of $x_i$

Example: genre identification in consumer behaviour analysis

-----

9. Causal modelling

__Instance:__

* a collection (dataset) $\mathbf{D}$ of datapoints from $\mathbf{X}$, e.g., $\mathbb{R}^m$

* a distinct dependent variable $x_i$

. . .

__Solution:__  a variable $x_j$ of $\mathbf{D}$ that controls $x_i$

__Measure:__  effectiveness of $x_j$ *tuning* to *tune* $x_i$ in turn.

. . .

Example: Exactly What food causes you to put on weight?

Controlled clinical trials, A/B testing.

<!-- --------- -->
# [Un]Supervision

## Supervised Data Science

* obtain a dataset of examples, inc. the  "target"  dimension, called *label*

* split it in training and test data

* run a. on the test data, find a putative solution

* test the quality/pred. power against test data

. . .

Regression involves a numeric target while classification
involves a categorical/binary one

-----

## Supervised

1: Regression

2: Classification

9: Causal Modelling

-----

## Could be either

3: Similarity matching,

7: link prediction,

8: data reduction

---

## (mostly) unsupervised

4: Clustering

5: co-occurrence grouping

6: profiling
