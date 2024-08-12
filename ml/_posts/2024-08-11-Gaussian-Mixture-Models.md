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
- mean ($$ \mu_i $$)

