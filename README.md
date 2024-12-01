# Colorizing Black and White Images with Conditional GANs

This project focuses on **colorizing black-and-white images to RGB** using a Conditional Generative Adversarial Network (cGAN). The goal is to generate realistic and vibrant color images from grayscale inputs.

## How It Works
- We use the **L-channel** from grayscale images as input to the generator model.
- The generator predicts the corresponding **ab-channels**, which are then combined with the original L-channel to create a fake RGB image.
- Both the real RGB image (ground truth) and the fake RGB image (from the generator) are passed to the discriminator model, which learns to distinguish between them.
- This adversarial training improves the generator’s ability to produce realistic colorizations.

## Dataset
We used the **COCO 2017 dataset** to train and evaluate the model.  
You can find the dataset on Kaggle: [COCO 2017 Dataset](https://www.kaggle.com/datasets/awsaf49/coco-2017-dataset).

## Features
- **Generator:** Learns to predict the ab-channels for colorization.
- **Discriminator:** Ensures the generated images are indistinguishable from real RGB images.
- **Loss Functions:** Combines adversarial loss and pixel-wise L2 loss to balance realism and accuracy.
- **Dataset Preprocessing:** Images resized to appropriate dimensions and normalized for efficient model training.

This project demonstrates how deep learning and cGANs can be leveraged to solve challenging computer vision tasks such as image colorization. 
