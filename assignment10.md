---
layout: default
title: Assignment 10
id: ass10
---


# Assignment 10: Self-Supervised Learning
**Deadline: December 20th, 9am**

**NOTE The deadline will likely be moved to January!**

**As usual, there's a notebook on E-Learning with some starter code.**

In this assignment, we want to explore self-supervised methods for extracting
features from unlabeled data, and then use those features for supervised tasks.

## General Pipeline

No matter the exact kind of model, we usually do something like this:
1. Define a self-supervised task, such as autoencoding, denoising, predicting
neighboring values, filling in blanks...
2. Build a network to solve the task. Often, this will be some kind of encoder-decoder
architecture.
3. Train the model.
4. Build a small "classification head" on top of your self-supervised model.
If that has an encoder-decoder structure, you will usually discard the decoder
   and put the classification head on top of the encoder.
5. Train the classification network on labeled data.


## Your task

For a dataset of your choice, implement the above pipeline. Try **at least two
different kinds** of self-supervised models; for each, train the model and then
use the features for a classification task.  
Also train a model directly on classification (no pre-training) and compare the
performance to the self-supervised models. Also compare the different self-supervision
methods with each other. 

To make these comparisons fair, your models should
have the same number of parameters. E.g. you might want to use the same "encoder"
architecture for each task, and add a small classification head on top; then, the
network that you train directly on classification should have the same architecture
as the encoder and the classification head combined.

The remainder of this text discusses some issues to keep in mind when
building autoencoders or similar models.


## Autoencoders in Tensorflow

Building autencoders in Tensorflow is pretty simple. You need to define an
encoding based on the input, a decoding based on the encoding, and a loss
function that measures the distance between decoding and input. An obvious choice
may be simply the mean squared error (but see below). To start off, you could try
simple MLPs. Note that you are in no way obligated to choose the "reverse"
encoder architecture for your encoder; e.g. you could use a 10-layer MLP as an
encoder and a single layer as a decoder if you wanted. As a start, you should
opt for an "undercomplete" architecture where the encoding is smaller than the
data.

**Note:** The activation
function of the last decoder layer is very important, as it needs to be able to
map the input data range. Having data in the range [0, 1] allows you to use a
sigmoid output activation, for example. Experiment with different activations
such as sigmoid or linear (i.e. no) activation and see how it affects the
model (do **not** use relu in the output layer!). Your loss function should also "fit" the output function, e.g. a sigmoid
output layer goes well with a binary (!) cross-entropy loss.

Note that you can use the Keras model APIs to build the encoder and decoder
as different models, which makes it easy to later use the encoder separately.
You can also have sub-models/layers participate in different models at the same time,
e.g. an `encoder` model can be part of an `autoencoder` model together with a `decoder`,
and of a classification model together with a `classifer_head`.


## Convolutional Autoencoders

Next, you should switch to a convolutional encoder/decoder to make use of the
fact that we are working with image data. The encoding should simply be one or
more convolutional layers, with any filter size and number of filters, plus
downsampling (strided convolutions or pooling). You can also
optionally apply fully-connected layers at the end. As an "inverse" of a
`Conv2D`, `Conv2DTranspose` is commonly used. However,
you could also use `UpSampling2D` along with regular convolutions.
 Again, there is no
requirement to make the parameters of encoder and decoder "fit", e.g. you don't
need to use the same filter sizes. However, you need to take care when choosing
 padding/strides such that the output has the same dimensions as the input. This
 can be a problem with MNIST (see the notebook).
  It also means that the last
convolutional (transpose) layer should have as many filters as the input 
space (e.g. one filter for MNIST or three for CIFAR).


## Other models

Even other self-supervised models are often similar to autoencoders.
For example, in a denoising
autoencoder, the input is a noisy version of the target (so input and target are not the same
anymore!), and the loss is computed between the output and this "clean" target.
The architecture can remain the same, however.  
Similarly, if the input has parts of the image removed and the task is to
reconstruct those parts, the target is once again the full image, but an autoencoding
architecture would in principle be appropriate once again.


## To freeze or not to freeze

Say, you trained some encoder network on a self-supervised task and now build
a classification head on top for labeled data. Now you want to train this model.
But _which parts_ do you actually train? It could be
1. Only the classification head, leaving the encoder untouched,
2. The full network including the encoder,
3. The classification head and some part of the encoder, say the last X layers...

There is a trade-off here:
- Allowing the encoder to be "fine-tuned" allows it to learn features that
are suited for classification, in case the self-supervised features are not optimal.
- However, this might cause the encoder to overfit on the training set. Training
only the classification head would keep the encoder features more general.
- The third option is a compromise between both.

Experiment! You can easily "freeze" layers or whole models by setting their 
`trainable` argument to `False`.


## Pre-training for low supervision scenarios

Self-supervised models are useful in that they can learn from unlabeled data. This can
significantly improve performance in settings where large amounts of data are
available, but few labels. We can artificially evoke such a situation by just 
"pretending" that parts of our data has no labels. Try this:

- Train a self-supervised model as before.
- Take a small random subset of the training data. Make sure it is actually random,
i.e. all labels are represented. You could go very low, e.g. 100 elements or so.
- Now train a classification net on only this small labeled subset!

As before, compare to a model that is trained directly on the classification task,
but only on the labeled subset. If everything works as expected, your self-supervised
model should outperform the directly trained one (on the test set)! This is because
the direct training massively overfits on the small dataset, whereas the self-supervised
model was able to learn features on all available data. You will most likely
want to freeze the encoder model, i.e. not fine-tune it -- if you did, the
self-supervised model would overfit, as well.
