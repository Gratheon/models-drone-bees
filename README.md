# 🐝 Bee-detection-ML2025
This repository presents a machine learning pipeline for **automatic detection of worker and drone bees from video frames**. The workflow focuses on handling severe class imbalance, large-scale image preprocessing, and comparative evaluation of modern object detection architectures, with **YOLOv10** selected for final experimentation.

### Repository Structure

The repository is organized into two main directories:

- **Model_selection** contains comparative experiments with multiple object detection architectures, including **YOLOv8**, **YOLOv10**, **YOLOv12**, and **RT-DETR**. These experiments are used to evaluate baseline performance under controlled preprocessing and augmentation settings.

- **project_beedetection** contains the full detection pipeline based on **YOLOv10**, including high-resolution image preprocessing, image tiling, dataset augmentation and balancing, manual dataset curation, and final model training and evaluation.

### Model Selection
Model selection experiments were conducted using a publicly available dataset from **Roboflow**. All images were resized to **640 × 640 pixels** using stretching prior to training. Data augmentation was applied by generating two augmented variants per image, including grayscale transformation applied to **15%** of samples and random noise added to up to **1.96%** of image pixels. The dataset was split into **training, validation, and test subsets** using a **40/5/5** ratio. The dataset is highly imbalanced, with a **drone-to-worker bee ratio of 34.3:1**.

### project_beedetection
Based on the results of the model selection stage, **YOLOv10** was selected for further development. A larger and more resource-rich dataset was sourced from the **Mississippi State University GRI Publications database**. Additional preprocessing steps were applied, including image tiling and manual dataset inspection. Multiple training runs were conducted on both imbalanced and balanced versions of the dataset to evaluate the effect of class balancing on detection performance.

Dataset source:
https://scholarsjunction.msstate.edu/gri-publications/4/

This project is released under the **MIT License**.
