# Skin Lesion Classification – IMA205 Project

This repository contains the implementation and report of the **IMA205 Medical Imaging Project** conducted at Telecom Paris. The goal of this project is to design and evaluate a complete pipeline for the **automated classification of skin lesion images** into eight categories (melanoma, basal cell carcinoma, vascular lesion...), contributing to the development of computer-aided diagnostic systems.

---

## Objectives

- Preprocess dermoscopic images to remove artifacts (hairs, black borders, illumination issues).  
- Segment lesions using adaptive thresholding and evaluate performance with the **Jaccard index**.  
- Extract clinically meaningful features following the **ABCD rule** (Asymmetry, Border, Color, Diameter) and texture descriptors.  
- Incorporate patient metadata (age, sex, lesion position) into the classification process.  
- Train and evaluate traditional classifiers (SVM, Random Forest) and deep learning models (CNNs).  
- Compare results with clinical expectations and published literature.  

---

## Dataset

The project relies on the **ISIC 2018/2019 challenge dataset**, which provides:  
- Training and test images of skin lesions.  
- Ground-truth segmentation masks for a subset of the data.  
- Metadata (patient age, sex, lesion location).  

---

## Methods

1. **Preprocessing**  
   - Brightness adjustment  
   - Hair removal using morphological filters (DullRazor)  
   - Black border removal  

2. **Segmentation**  
   - Adaptive thresholding on color channels  
   - Morphological cleaning and contour extraction  
   - Accuracy evaluation with the Jaccard index  

3. **Feature Extraction**  
   - Asymmetry measures and centroid distance  
   - Border irregularities via convex hull comparison  
   - Color statistics in RGB and HSV channels  
   - Diameter estimation from minimum enclosing circle  
   - Texture descriptors using Gray-Level Co-occurrence Matrix (GLCM)  

4. **Classification**  
   - Feature-based classification with SVM and Random Forest  
   - End-to-end training of Convolutional Neural Networks (CNNs)  


