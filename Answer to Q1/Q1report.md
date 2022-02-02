---
title : Answer to Q1
author: Ziyuan Wang
fontsize: 11.5pt
bibliography: [../../../../Refs/NTU_tasks.bib]
output: pdf_document
---

## Main point of that paper

In this paper, a machine learning model for predicting tumor purity from H&E stained histopathological sections was developed, thus making predictions consistent with genomic tumor purity values. This approach is less costly and time consuming than genome sequencing.

*The image input is regarded as a bag and the $1mm^2$ regions are considered as instances*

## Implementations and Results

### The aim of the model

![Overview of the paper](https://www.biorxiv.org/content/biorxiv/early/2021/07/09/2021.07.08.451443/F1.large.jpg?width=800&height=600&carousel=1)

+ Multi-instance learning (MIL) was carried out through the input sample image, and the tumor nuclear purity of the sample was predicted by bag-level feature vector in the output layer.
+ Obtain a spatial tumor purity map for a slide showing every $1mm^2$ region purity.
+ Hierarchical clustering was performed using only weak labels to obtain features that could distinguish cancer tissue from normal tissue.
+ Classify samples into tumor vs. normal.

### Architecture

**Modules**

+ Feature Extractor module
+ MIL pooling filter module
+ bag-level representation transformation

ResNet18 model as the feature extractor module and a three-layer multilayer-perceptron as the bag-level representation transformation module.

Unlike max/min-pooling which converts each dimention of extractor features different instances into one value, MIL pooling filter module converts them into a distribution using 21 sample points(Default). In this paper, the performance of this pooling method is better than maximum pooling, minimum pooling and average pooling.

### Performance

![TCGA dataset benchmark](https://www.biorxiv.org/content/biorxiv/early/2021/07/09/2021.07.08.451443/F2.large.jpg?width=800&height=600&carousel=1)

This tool has a high correlation with the results obtained from transcriptome determination of tumor purity in samples, although there are some outliers.

![Comparasion with other tools](https://www.biorxiv.org/content/biorxiv/early/2021/07/09/2021.07.08.451443/F3.large.jpg?width=800&height=600&carousel=1)

AUC value (0.991) was utilized to evaluate our model perform via, which tumor samples were separated from normal samples in LUAD cohort. Classical image processing and machine learning-based method[@Yu2016a] and the DNA plasma-based method[@Sozzi2003] (0.85,0.94 respectively). Other models, such as, the deep learning model of [@Coudray2018a] (AUC: 0.993) ans [@Fu2020a] (AUC: 0.977 with 95% CI: 0.976 - 0.978). However, there is one concern about the dataset preparation methods of @Coudray2018a and @Fu2020a. How they sampled the data made their models’ performance illusory

## Discussion

Advantages:

+ Weak tumor purity labels necessitated a MIL approach. Pixel-level annoations(expensive) can be avoided
+ Complement spatial-omics(scRNA-seq) which can be seen from the Fig1.

Source of the error:

+ Lack of samples.
+ Histopathology slides from different areas.
+ Some limitation of H&E ained histopathology slides.

## Reference