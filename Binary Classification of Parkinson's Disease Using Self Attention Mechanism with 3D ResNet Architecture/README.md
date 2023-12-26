# Overview
This repository contains the code and documentation for a project focused on the classification of Parkinson's Disease using advanced image analysis techniques. The project leverages 3D complex medical 3T structural MRI scans in the form of NIfTI files. Despite the small dataset size, the images are characterized by their complexity and high resolution.
# Project Structure
* **Data:** The dataset consists of NIfTI files containing 3D structural MRI scans.
* **Preprocessing:** OpenCV and ANTsPy were employed for preprocessing, ensuring that the input data is suitable for neural network training.
* **Augmentation:** TensorFlow's dataloader was utilized for data augmentation to enhance model generalization.
## Methodology
The primary focus of the project was to investigate the performance of ResNet architectures on high-resolution medical images with limited instances. Specifically, ResNet-18 and ResNet-34 were employed to explore how well these architectures could handle the unique challenges posed by small datasets with intricate imaging.

### Residual Neural Network (ResNet) Architecture
**ResNet-18 and ResNet-34:**  Both architectures demonstrated effectiveness in extracting features from high-resolution medical images without suffering from vanishing gradient descent. The models showcased promising results in terms of classification accuracy and robustness.

**ResNet-101:** This deeper architecture exhibited instability in the results, indicating that a more extensive network did not necessarily lead to improved performance with the given dataset.
### Attention Mechanism Integration
# Conclusion
The experimentation with ResNet architectures on high-resolution medical images yielded valuable insights into their applicability for Parkinson's Disease classification. The use of attention mechanisms contributed to refining the model's attention and feature extraction, potentially improving its diagnostic capabilities.
