---
layout: post
title: Gaussian Mixture Models
description: >
  Mathematical formula (from statistics and math perspective) and python implementation for different types of regression models
# image: 
#   path: /assets/img/blog/recommend.jpg
#   srcset:
#     1060w: /assets/img/blog/recommend.jpg
#     530w:  /assets/img/blog/recommend_50.jpg
#     265w:  /assets/img/blog/recommend_25.jpg
sitemap: false
hide_last_modified: true
---

# Gaussian Mixture Models

* toc
{:toc .large-only}

## What is Gaussian Mixture Models (GMM)?

A Gaussian Mixture Models (GMM or Mixture of Gaussians) is a probabilitsitc model used to represent a mixture of multiple Gaussian distributions. 

## Motivation
K-means algorithm would typically assign each data to exactly one cluster, but what if these clusters are overlapping. In this case, we do not have enough information to tell which cluster are right or which cluster are dominating. In Gaussian mixture models, clusters are modeled as gaussian components which has its own mean and variance. In this case, we could assign each data to cluster with some probability. This approache gives probability model of data ("generate").

## Concepts of GMM
#### Responsibility ($$\gamma_{ij}$$)

$$
\gamma_{ij} = \frac{\pi_i \mathcal{N}(x_j \mid \mu_i, \Sigma_i)}{\sum_{k=1}^K \pi_k \mathcal{N}(x_j \mid \mu_k, \Sigma_k)}
$$

#### Mean ($$ \mu_i $$)

The mean vector for each Gaussian component $$ i $$, representing the center of the Gaussian distribution.

$$
\mu_i = \frac{\sum_{j=1}^N \gamma_{ij} x_j}{\sum_{j=1}^N \gamma_{ij}}
$$


#### Covariances ($$ \Sigma_i $$)
The covariance matrix for each Gaussian component, representing the spread and orientation of the Gaussian distribution in the feature space.

$$
\Sigma_i = \frac{\sum_{j=1}^N \gamma_{ij} (x_j - \mu_i)(x_j - \mu_i)^T}{\sum_{j=1}^N \gamma_{ij}}
$$


#### Weights ($$ \pi_i $$)

The mixing coefficients representing the weight of each Gaussian component in the mixture.

$$
\pi_i = \frac{\sum_{j=1}^N \gamma_{ij}}{N}
$$

## Fitting a GMM

### Initialization

### Expectation-Maximization (EM) Algorithm
We use Expectation-Maximization (EM) algorithm to itera

#### Expectation Step (E-Step)
Calculate the probability that each data point belongs to each Gaussian component. We use the current estimates of the means, covariance, and weights.


#### Maximization Step (M-Step)
Update the parameters of the Gaussian components (means, covariances, and weights) to maximize the likelihood of the data given the current probabilities.

#### Covergence:
The algorithm iterates betwee nthe E-Step and M-Step until convergence, which is typically when chagnes in the log-likelihood or parameters are below a certain threshold.
