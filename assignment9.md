---
layout: default
title: Assignment 9
id: ass9
---


# Assignment 9: Introspection
**Deadline: December 13th, 9am**


In this assignment, we will implement gradient-based model analysis both for
 creating saliency maps (local) and for feature visualization (global). 
 You can  also take inspiration from the 
[DeepDream tutorial](https://www.tensorflow.org/tutorials/generative/deepdream).
It is recommended that you work on image data as this makes visual inspection of
the results simple and intuitive.

You are welcome to use pre-trained ImageNet models from the `tf.keras.applications`
module. The tutorial linked above uses an Inception model, for example. Note
that these models generally expect rather large inputs, for example 224x224 pixels.
However, you can generally take arbitrary images and resize them (e.g. `tf.image.resize`)
to the necessary dimensions. Also note that these models generally require specific
pre-processing of the input; all modules have their own functions for this, see the API.

You can of course
also train your own models on CIFAR or something similar. Smaller data and models
usually makes everything much faster. :) You can also prototype on CIFAR and then
try to generalize to bigger images/models once everything works.

## Gradient-based saliency map (sensitivity analysis)

Run a batch of inputs through the trained model.
Wrap this in a GradientTape where you watch the input batch
(batch size can be 1 if you'd like to just produce a single saliency map).
and compute the gradient for a particular logit or its softmax output _with 
respect to the input_.
This tells us how a change in each input pixel would affect the class output.
This already gives you a batch of gradient-based saliency maps!
Plot the saliency map next to the original image or superimpose it.
Do the saliency maps seem to make sense? How would you interpret them?

Note that you will get _very_ different results when using logits or softmax probabilities!
  - Logits basically only consider the "evidence" for a certain class.
  - Softmax outputs consider _all_ classes, as the probability can be increased by
  increasing the respective class logits _or_ reducing logits for other classes!
  - Softmax tends to saturate, leading to bad gradients and thus unclear saliency maps.
  - You can also use the _log probabilities_; these are best computed via the built-in
  cross-entropy losses: By supplying the outputs along with the desired labels, and
  _minimizing_ this loss, we are actually maximizing the log probabilities for those labels.

Further notes:
- It makes sense to take the sign of the gradient into account when 
interpreting them.
Negative gradients indicate a decrease in output value, positive 
gradients an increase. This means you should use a _diverging_ colormap with 
  separate colors for positive and negative values for plotting.
- Alternatively, maybe using absolute values of the gradient and a _sequential_ colormap might
  make more sense. What do you think?
  - Yet another alternative could be to rectify the filter maps, i.e. replace negative
  values by 0. This only leaves the gradients that would _increase_ the class
  probability.
  - See [here](https://matplotlib.org/stable/users/explain/colors/colormaps.html)
  for information on colormaps!
- When using color images, gradients are also multi-channel. These are then usually
averaged over the channel axis to produce single-channel saliency maps.
- You can try smoothing the saliency maps, e.g. with a gaussian filter. This will
generally make them look "better", but also falsifies the actual information somewhat.

Saliency maps can be especially interesting for wrongly classified inputs. Here, you
could compute saliency maps for either the _correct_ class, or the _predicted_ one.
How do the two differ?


## Activation Maximization

Extend the code from the previous part to create an optimal input for a 
particular class.

- Start with a _randomly initalized image_, not one from the dataset (although you _could_ also
use a dataset image as a starting point).
- Multiply the gradients with a small constant (like a learning rate) and add them
to the input.
- Repeat this multiple times, computing new gradients with respect to the input each
time.
Essentially, you are writing a "training loop" for producing an optimal input for
a certain class (do _not_ train the model weights!).  

**Note:** You need to take care that the optimized inputs actually stay valid images
throughout the process, e.g. by clipping to [0, 1] after each gradient step, or by
using a sigmoid function to produce the images.


Does the resulting input look natural?
How do the inputs change when applying many steps of optimization?
How do the optimal inputs differ when initializing the optimization with random 
noise instead of real examples?
Can you see differences between optimizing a logit or a (log) softmax probability?

**Bonus**: Apply regularization strategies to make the optimal input more 
natural-looking.
You can also optimize for _hidden features_ of the network (instead of outputs)
assuming you can "extract" them from the model you built. Distill has 
[an article](https://distill.pub/2017/feature-visualization/) that can provide
some inspiration.


## Bonus: Unmasking "Clever Hans" Predictors

Creating saliency maps, and then not doing anything with them, might seem slightly
pointless. If you have time, you can try the following experiment:

- For a given dataset (e.g. MNIST or CIFAR), pick one or more classes and add simple
"identifiers" to each image. E.g. maybe all dogs in CIFAR get a bright red square
in the lower left corner.
- Train a model on this modified data. It should have a particularly simple time
on the modified class(es).
- Compute saliency maps for some images of the class(es) in question. Do they indicate
that the model is looking at the markers, or does it still appear to do classification
based on "correct" features?


## Submission

Include code for creating saliency maps as well as activation maximization.
Show some comparisons/experiments for various inputs, concerning e.g. the different
ways to represent saliency maps (smoothed or not, thresholded or not, etc.).
Also show examples of activation maximization for various classes (e.g. CIFAR cars
vs. dogs vs. ships etc...). In case you applied regularization techniques, also
document how those influenced the results!
