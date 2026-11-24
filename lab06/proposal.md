# Lab 06 Individual Project Proposal

**Student Name:** Maria Namatovu  
**Student ID:** 70897

## Problem Statement

Build a supervised machine-learning URL classifier that estimates whether a URL is benign or suspicious. Most suspicious URL appear genuine to the human eye - and yet they're actually malicious. The objective is to build a model which will use URL-only and domain-derived lexical features, allowing fast inference and flagging the URL if it is malicious. The end goal is to have the model have fewer number of false positives and false negatives and higher number of true positives. One potential usecase of this would integrating the model inside a browser extension for fast URL scanning.

## Dataset

1. PhishTank is a dataset that contains community-submitted and verified phishing URLs.
2. Tranco is a dataset that contains benign URLs samples with popular domains included


## Features
The model will be trained to use the following features during training:
URL Length, IP address for domains, number of sub-domains, hostname length, number of query parameters, presence of suspicious delimiters such as `@` or excessive backslashes

## Success Metric

Primary success will be minimizing false-positive  and false negative outcomes and maximizing true positives

## Deployment Constraints

Latency must be ≤ 50 ms and should include URL ingestion, normalization, feature extraction, and model inference. For a browser deployment - the inference should happen without blocking navigation in the browser.

## Planned Optimization

The optimization technique will be feature selection followed by quantization to improve model inference - and since the model of choice will be initially a logistic regression model - this will reduce model size and improve inference performance. Features that add little validation value will be removed, and the final compact model will be exported to a browser-compatible format  with a size < 5MB
