
# MLOps_88
The goal of this project is to classify chest X-ray images into four categories: Normal, Pneumonia, COVID-19, and Tuberculosis. The aim is not only to obtain good classification performance, but also to structure the project according to MLOps principles, including reproducible experiments, modular code, and a clear training and evaluation pipeline.

The project will be implemented using the PyTorch together with the MONAI framework, which provides domain-specific tools and model architectures for medical image analysis. The framework will be integrated into a structured project setup that separates data handling, model definition, training, and evaluation.

The dataset that we will use in this project is the Chest X-Ray (Pneumonia, COVID-19, Tuberculosis) dataset from Kaggle, which contains 7 135 chest X-ray images divided into training (6326), validation (38), and test sets (771). The images are organized into four classes and represent a realistic medical imaging scenario with inter-class similarities and class imbalance.

During the project we will look at convolutional neural networks (CNNs) for image classification. In particular, CNN architectures provided by the MONAI framework, such as ResNet and DenseNet, are used and fine-tuned on the chest X-ray dataset. These architectures are widely adopted in medical image analysis and provide strong baseline performance while keeping the model complexity manageable.
