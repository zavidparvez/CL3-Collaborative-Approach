# Transfer Learning, Federated Learning, and Incremental Learning with Concept Drift Using X-ray Images

This project implements Transfer Learning, Federated Learning, and Incremental Learning techniques with a focus on Concept Drift using X-ray image datasets. The goal is to incrementally improve a model using federated data from multiple clients without centralizing the data, which is crucial for privacy-sensitive applications such as medical imaging.

## Table of Contents
- [Project Overview](#project-overview)
- [Installation](#installation)


## Project Overview

- **Transfer Learning**: A pre-trained EfficientNetB7 model is fine-tuned on X-ray images to perform classification.
- **Federated Learning**: Local models on different clients are trained on decentralized X-ray data, and the results are aggregated to improve the global model.
- **Incremental Learning with Concept Drift**: The model adapts incrementally to new data, addressing the concept drift phenomenon in X-ray images over time.

## Installation

Clone the repository:

- **git clone https://github.com/zavidparvez/CL3-Collaborative-Approach.git
- **cd CL3-Collaborative-Approach

Installation
Prerequisites
Ensure you have the following installed:

- **Python 3.7 or higher
- **TensorFlow 2.x
- **Keras
- **EfficientNet

Install the required Python packages by running:
- **pip install -r requirements.txt


