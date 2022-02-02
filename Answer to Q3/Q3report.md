---
title : Answer to Q1
author: Ziyuan Wang
fontsize: 11.5pt
bibliography: [../../../../Refs/NTU_tasks.bib]
output: pdf_document
# bibliography: [../../../../Refs/NTU_tasks.bib]

---

# Answer to Q3

## Brief Summary of the paper [@Wang2021d]

### What is doppelgänger effects

Training and validation sets are highly similar for accidental or other reasons[@Wang2021d].In many cases, the model generalization ability of training is poor due to the influence of the split body effect, and the performance is poor when the data is not learned. The current methods for identifying and improving the doppleganger effect are not universal enough and need to be improved. This paper discusses the popularity of functional doppleganger of biomedical data, the influence of data doppleganger on ML, and the methods to reduce the doppleganger effect.

### Abundance of data doppelgänger in biomedical data

Some data doppelgänger pointed out in the article

+ protein function prediction[@Wass2008;@Friedberg2006]. *The functions proteins with less similar sequences but similar functions cannot be predecited*
+ quantitative structure–activity relationship (QSAR) models[@Paul2021]. *The QSAR model assumes that molecules with similar structures have similar activities and will encounter problems when facing molecules with different structures.*

Other data doppelgänger in bioinformatics

+ Single-cell RNA-seq/ATAC *Many single-cell machine learning analyses rely on data sets obtained in similar sequencing environments, similar donor sampling, and other conditions, which can lead to some misleading analyses.Batch effects in different data sets can also interfere with predictions.[@Luecken2021]*
+ Sequence Analysis *Sequence analysis is based on the assumption that DNA sequences with similar sequences have similar functions, which is the same as data doppelgänger in the paper.*

### PPCC applied in datasets
![**PPCC-Fig**](https://ars.els-cdn.com/content/image/1-s2.0-S1359644621004554-gr3.jpg)

The pairwise Pearson’s correlation coefficient (PPCC) can capture the relation between different samples if the value is high we can deduce data doppelgängers.



When using PPCC to judge data doppelgängers, we can see from the figure that, in general, more doppelgängers will interfere with the machine learning discriminator. As the number of duplicates in the Validation dataset increases, the accuracy of machine learning will mistakenly increase.

### Ameliorate data doppelgängers

Current methods:

+ Analyze the specific context of the data to develop a more comprehensive and rigorous assessment strategy[@Cao2019].
+ Delete PPCC data clone. However, it needs to lose too many samples[@Ma2018;@Lakiotaki2018].

Challenges : **How to solve data doppelgängers without significantly reducing data.**


## Whether doppelgänger effects are unique to biomedical data
No

One example is Face recognition. [@Rathgeb2022]The recognition accuracy of doppelgänger effects was improved significantly after the problem was solved.

## Avoid doppelgänger effects in biomedical data

Using generative model to extract embeddings[@Luecken2021;@Lotfollahi2019;@Seninge2021].For scRNA-seq, data doppelgängers can be significantly reduced by using self-supervised models, and we can explore potential relationships between different gene expressions through hidden layers. When our model is used to predict the samples of new donations, even if the effect is caused by batch effect or the difference in physical condition between people from different regions due to different reasons, we can still capture a lot of effective information and make further prediction.

For chemical structure data, we can adopt embedding, including graph neural network, graph convolution neural network and other methods, which have been proved in some studies. In these neural networks, the model learns the relationship of edges (chemical bonds) between different types of nodes (representing chemical elements or groups). In this case, the model is very generalizing and can predict chemical structures that are not present in the data set [@Ding2021;@Stokes2020].

## Reference