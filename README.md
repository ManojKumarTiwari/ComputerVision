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
  - https://paperswithcode.com/
  - https://modelzoo.co/
  - https://modeldepot.io/
  
| Dataset                              | Description |
|--------------------------------------|-------------|
| [ImageNet](http://www.image-net.org) |             |
| [Tensorflow Datasets](https://www.tensorflow.org/datasets/catalog/overview)                         |             |

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
      
# Local Setup
  - [Lambda Stack](https://lambdalabs.com/lambda-stack-deep-learning-software)
  - [ROCm](https://rocmdocs.amd.com/en/latest/) - Setup of AMD GPU
      
# Free Online Notebooks
  - Colab
  - [Binder](https://mybinder.org/)
  - IBM Watson Studio
  - Gradient
  
# Cloud Visual Recognition APIs
  - [Microsoft Cognitive Services](https://azure.microsoft.com/en-in/services/cognitive-services/)
  - [Google Cloud Vision](https://cloud.google.com/vision)
  - [Amazon Rekognition](https://aws.amazon.com/rekognition/)
  - [IBM Watson Visual Recognition](https://cloud.ibm.com/catalog/services/visual-recognition)
  - [Clarifai](https://www.clarifai.com/)
  - [Algorithmia](https://algorithmia.com/)
  
# Tools
  - Tensorflow Datasets - for most optimize implementation
  - Tensorboard - to visulize many aspects of training
  - [What-If Tool](https://pair-code.github.io/what-if-tool/) - to compare models
  - tf-explain - Analyze decisions made by the network
  - Keras Tuner - automatic tuning of hyperparameters in tf.keras
  
  - [For Any Question First Check this out](https://github.com/PracticalDL/Practical-Deep-Learning-Book/tree/master/code/chapter-7)
  
  - [Hyperas](https://github.com/maxpumperla/hyperas) - Hyperparameter tunner
  - [Hyperopt](https://github.com/hyperopt/hyperopt) - Hyperparameter tunner
  - [BayesianOptimization](https://github.com/fmfn/BayesianOptimization) - Hyperparameter tunner
  - AutoKeras - Automates Neural Architecture Search (NAS) across different tasks like image, text, and audio classification and image detection
  - AdaNet - NAS
  - AutoAugment - Utilizes reinforcement learning to improve the amount and diversity of data in an existing training dataset, thereby increasing accuracy
  - TensorFlow Debugger - for debugging
  - nvidia-smi - This command shows GPU statistics including utilization.
  - TensorFlow Profiler + TensorBoard - This visualizes program execution interactively in a timeline within TensorBoard.
  - OpenCV - Data Augmentation
  - Pillow - Data Augmentation
  - [Knock Knock](https://github.com/huggingface/knockknock) - Get notified when your training ends with only two additional lines of code
  - [Fast Progress Bar](https://github.com/fastai/fastprogress) - Simple and flexible progress bar for Jupyter Notebook and console
  - [Netron](https://github.com/lutzroeder/netron) - Visualizer for neural network, deep learning and machine learning models
  - [NN-SVG](https://alexlenail.me/NN-SVG/) - for making neural networks diagrams
  - [PlotNeuralNet](https://github.com/HarisIqbal88/PlotNeuralNet) - Latex code for making neural networks diagrams
  - [Fatkun Batch Download Image](https://chrome.google.com/webstore/detail/fatkun-batch-download-ima/nnjjahlikiabnchcpehcpkdeckfgnohf?hl=en) - Chrome Extension to Download images
  
      
# Some Important Steps
  - Class Activation Map
  - Transfer Learning
  - Fine Tuning
  - Data Argumentation
  - Most Confident/Least Confident/ Incorrect with high Confidence
  - Feature Extraction (feature vectors or embeddings or bottleneck features)

# Deployment Platforms
  - Computer/Server
    - HTTP servers
      - Flask, Django
    - Hosted and managed cloud stacks
      - Google Cloud ML
      - Azure ML
      - Amazon Sage Maker
    - Manually managed serving libraries
      - TensorFlow Serving
        - [Using Docker](https://www.tensorflow.org/tfx/serving/docker)
      - NVIDIA TensorRT
    - Cloud AI orchestration frameworks
      - KubeFlow
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
    2. Mastering OpenCV 4 with Python
  
# Github
  For book 1: https://github.com/PracticalDL/Practical-Deep-Learning-Book
  For book 2: https://github.com/PacktPublishing/Mastering-OpenCV-4-with-Python
  
# Useful Tools and Links
- https://projector.tensorflow.org/ - Embedding Projector
- http://ann-benchmarks.com/ - benchmarking environment for approximate nearest neighbor algorithms search
- https://www.slideshare.net/anirudhkoul/deep-learning-on-mobile-2019-practitioners-guide - Presentation on DL on Mobile 2019

# To Do
1. Build a Web App that takes an image and outputs whether that image is horse or human
2. Build a Web App that takes an image and produces similar images 
