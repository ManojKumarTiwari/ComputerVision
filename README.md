# ComputerVision
Everything about Computer Vision

# For Beginners CNN in Action
  - Tensorflow Playgroud
  - Andrej Karpathy’s ConvNetJS

# Objects (Inputs)
  - Image
  - Video
  
# Types of Computer-Vision Tasks
  - Classification
    - Binary Classification
    - Multiclass Classification
  - Localization
  - Detection
  - Segmentation
    - Semantic Segmentation
    - Instance-based Segmentation
    
# Model Zoo
  - https://keras.io/api/applications/
  - https://github.com/GoogleCloudPlatform/keras-idiomatic-programmer/tree/master/zoo
  
# Dataset

    | Dataset                              | Description |
    |--------------------------------------|-------------|
    | [ImageNet](http://www.image-net.org) |             |
    |             Caltech                         |             |
    | Tensorflow Datasets                         |             |
    
# Model Architecture
    
    | Model Architecture |
    |--------------------|
    | Resnet             |
    
# Approaches for training/inference 
  - Cloud Based API (inference)
  - Pretrained Model (inference)
  - Cloud Based Model Traning (traning/inference)
  - Custom Training (training/inference)
    - ## Requirements
      - Dataset
      - Model Architecture
      - Framework
      - Hardware
      
# Tools
  - Tensorflow Datasets - for quick implementation
  - Tensorboard - to visulize many aspects of training
  - What-If Tool - to compare models
  - tf-explain - Analyze decisions made by the network
  - Keras Tuner - automatic tuning of hyperparameters in tf.keras
  - AutoKeras - Automates Neural Architecture Search (NAS) across different tasks like image, text, and audio classification and image detection
  - AutoAugment - Utilizes reinforcement learning to improve the amount and diversity of data in an existing training dataset, thereby increasing accuracy
      
# Some Important Steps
  - Class Activation Map
  - Transfer Learning
  - Fine Tuning
  - Data Argumentation
  - Most Confident/Least Confident/ Incorrect with high Confidence
  - Feature Extraction (feature vectors or embeddings or bottleneck features)

# Platforms
  - Computer/Server
  - Website (Browser)
  - Mobile
    - iOS
    - Android
  - Edge Device
    - Raspberry Pi
    - Jetson
    
# Applications
- Reverse Image Search Engine (Instance Retrieval)
  - Similarity Search
    - K-Nearest Neighboors
    - Approximate Nearest Neighbors
      - Annoy
      - FLANN
      - Faiss
      - NGT
      - NMSLIB
- Siamese Networks for One-Shot Face Verification

# Sources
  # Books
    1. Practical Deep Learning for Cloud, Mobile, and Edge
  
# Github
  For book 1: https://github.com/PracticalDL/Practical-Deep-Learning-Book
  
# Useful Tools and Links
- https://projector.tensorflow.org/ - Embedding Projector
- http://ann-benchmarks.com/ - benchmarking environment for approximate nearest neighbor algorithms search
- https://www.slideshare.net/anirudhkoul/deep-learning-on-mobile-2019-practitioners-guide - Presentation on DL on Mobile 2019

# To Do
1. Build a Web App that takes an image and outputs whether that image is horse or human
2. Build a Web App that takes an image and produces similar images 
