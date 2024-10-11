This repository provides an implementation of X-ray image classification using a combination of Transfer Learning, Federated Learning, and Incremental Learning to address Concept Drift. The project focuses on applying federated learning techniques across multiple clients (e.g., hospitals or institutions) using pre-trained models to minimize the need for sharing sensitive data, while also adapting to changing data patterns over time.

Table of Contents
Project Overview
Project Structure
Installation
Running Experiments
Model Descriptions
Transfer Learning
Federated Learning
Incremental Learning and Concept Drift
Contributing
License
Project Overview
This project applies state-of-the-art machine learning techniques to classify X-ray images in a distributed learning environment. By leveraging transfer learning from pre-trained models like EfficientNet and federating the learning process across multiple clients, the system avoids direct data sharing and instead aggregates learning updates. Furthermore, incremental learning is employed to adapt to Concept Drift — changes in data distributions over time — to ensure continuous model improvement.

Key Components:

Transfer Learning: Fine-tune a pre-trained model (EfficientNet) to perform X-ray classification.
Federated Learning: Distribute the learning process across five clients, aggregating weights to create a global model.
Incremental Learning with Concept Drift Detection: Update the model dynamically in response to changes in data distribution.
