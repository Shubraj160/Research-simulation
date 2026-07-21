# Federated Learning Research Simulation

This repository contains research-oriented implementations of classical **Federated Learning (FL)** algorithms developed using **Python, TensorFlow, and Keras**. The project evaluates the performance of **FedAvg, FedProx, and SCAFFOLD** under both **IID** and **Non-IID** data distributions with varying client participation rates (100%, 50%, and 30%). To efficiently emulate a federated environment on a single machine, **Python multithreading (`ThreadPoolExecutor`)** is used to simulate the concurrent execution of **10 virtual clients**, where each thread represents an independent client performing local training before server-side aggregation. 

## Features
- Implementation of FedAvg, FedProx, and SCAFFOLD
- IID and Non-IID data partitioning
- Configurable client participation rates (100%, 50%, and 30%)
- Simulation of **10 virtual clients** using Python multithreading
- CNN-based experiments on the MNIST dataset
- Performance evaluation using training loss and test accuracy

## Technologies Used

| Component | Technology |
|-----------|------------|
| Programming Language | Python 3.x |
| Deep Learning Framework | TensorFlow |
| High-Level API | Keras |
| Numerical Computing | NumPy |
| Data Handling | Pandas |
| Visualization | Matplotlib |
| Parallel Simulation | `concurrent.futures.ThreadPoolExecutor` |
| Dataset | MNIST |



## Experimental Configuration

- **Dataset:** MNIST
- **Model:** Convolutional Neural Network (CNN)
- **Optimizer:** SGD
- **Local Epochs:** 2
- **Virtual Clients:** 10
- **Client Participation:** 100%, 50%, 30%
- **Data Distribution:** IID and Non-IID
- **Evaluation Metrics:** Test Accuracy, Training Loss

## Objective

The objective of this repository is to compare the behavior of classical federated learning optimization algorithms under different data heterogeneity and client participation settings while demonstrating the use of multithreading to efficiently simulate a federated environment with **10 virtual clients** on a single machine.

