---
layout: default
title: Reading Assignment
id: reading8
---


# Reading Assignment: Introspection
[The blog about tf-explain](https://www.sicara.ai/blog/2019-08-28-interpretability-deep-learning-tensorflow) gives a nice overview about different introspection techniques but is not very comprehensive.  
The toolbox works with TF 2.0, but currently only works on image data.

[This overview blog post](https://distill.pub/2017/feature-visualization/)
and a [more recent addition](https://distill.pub/2018/building-blocks/) will give you a nice overview about feature visualization.

Now it's time to look at some specific techniques. Dig as deep as you like! These websites (and papers) provide plenty of background information.

**Beginnings of the Field**
- Basic Saliency Maps and Feature Visualization in   
  [Simonyan et al. (2013) Deep Inside Convolutional Networks](https://arxiv.org/pdf/1312.6034.pdf)

**Saliency Maps**
- [GradCAM](http://gradcam.cloudcv.org/) Gradient-weighted Class Activation Mapping
- [Heatmapping](http://heatmapping.org/), especially [Montavon et al. (2017) Methods for Interpreting and Understanding Deep Neural Networks](https://arxiv.org/abs/1706.07979)
- [Sanity Checks for Saliency Maps](https://proceedings.neurips.cc/paper_files/paper/2018/file/294a8ed24b1ad22ec2e7efea049b8737-Paper.pdf)

**Feature Visualization**
- [Deepvis](http://yosinski.com/deepvis)
  (code and paper are linked)

**Representation Analysis**
- [Linear Classifier Probes](https://arxiv.org/pdf/1610.01644.pdf)
- [Visualizing NNs with the Grand Tour](https://distill.pub/2020/grand-tour/)
- [Visualizing Deep Neural Networks with Topographic Activation Maps](https://ebooks.iospress.nl/volumearticle/63328)  
  *This one is our work ;)*


## Optional

**Some Architecture-Specific Approaches**

- [LSTMVis](http://lstm.seas.harvard.edu/)
	(code and paper are linked)
- [exBert - Visual Analysis of Transformer Models](https://exbert.net) (code and paper are linked)
- [Abnar & Zuidema (2020) Quantify Attention Flow in Transformers](https://arxiv.org/abs/2005.00928)

**Overview Literature about Interpretable AI**
- [Christoph Molnar - Interpretable ML Book](https://christophm.github.io/interpretable-ml-book/)
- Nauta et al. (2023) From Anecdotal Evidence to Quantitative Evaluation Methods: A Systematic Review on Evaluating Explainable AI  
    [published version](https://dl.acm.org/doi/10.1145/3583558), [arXiv version including supplementary material](https://arxiv.org/abs/2201.08164) 
