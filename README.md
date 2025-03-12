# Stackelberg Game-Based Data Offloading for Edge Networks

This repository contains the implementation code for the paper "Stackelberg Games for Data Offloading in Edge Networks" by Anurag Bhattacharjee, Adrija Karmakar, Aditi Jalan, Anjan Bandyopadhyay.

## Overview

This project implements a Stackelberg game-theoretic framework for optimizing data offloading decisions in edge computing environments. The framework positions mobile devices as leaders making strategic offloading decisions, while edge servers act as followers allocating computational resources. The implementation demonstrates significant improvements in energy efficiency, latency reduction, and resource utilization compared to traditional offloading methods.

## Key Features

- Implementation of a Stackelberg game-based data offloading algorithm
- Simulation of mobile device and edge server interactions
- Evaluation of multiple performance metrics including:
  - Energy consumption
  - Latency
  - Mobile device utility
  - Edge server utility
  - Offloading rate
  - Resource utilization
- Comparison with baseline methods (local execution, random offloading, greedy offloading)
- Parameter sensitivity analysis

## Performance Results

Our approach demonstrates significant improvements across all metrics:

| Metric | Initial Value | Final Value | Improvement (%) |
|--------|--------------|-------------|-----------------|
| MD Utility | 54.76 | 99.66 | 82.01 |
| ES Utility | 154.76 | 199.66 | 29.02 |
| Energy Consumption (J) | 96.19 | 60.27 | 37.35 |
| Latency (ms) | 280.97 | 101.35 | 63.93 |
| Offloading Rate (%) | 35.71 | 89.60 | 150.90 |
| Resource Utilization (%) | 44.76 | 89.66 | 100.33 |

Compared to baseline methods:

| Method | Energy (J) | Latency (ms) | MD Utility |
|--------|------------|--------------|------------|
| Local Execution | 120.00 | 250.00 | 60.00 |
| Random Offloading | 90.00 | 180.00 | 75.00 |
| Greedy Offloading | 75.00 | 130.00 | 85.00 |
| Stackelberg | 60.27 | 101.37 | 99.66 |

## Requirements

- Python 3.7+
- NumPy
- Matplotlib
- Pandas
- Seaborn

## Installation

