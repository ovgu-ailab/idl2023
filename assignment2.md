---
layout: default
title: Assignment 2
id: ass2
---


# Assignment 2: Let the Tensors Board?
**Deadline: October 25th, 9am**

Visualizing the learning progress as well as the behavior of a deep model is
extremely useful (if not necessary) for troubleshooting in case of unexpected
outcomes (or just bad results). In this assignment, you will get to know
TensorBoard, Tensorflow's built-in visualization suite, and use it to diagnose
some common problems with training deep models. **Note:** TensorBoard seems to
work best with Chrome-based browsers. Other browsers may take a *very* long
time to load, or not display the data correctly.


## Datasets

It should go without saying that loading numpy arrays and taking slices of
these as batches (as we did in the last assignment) isn't a great way of
providing data to the training algorithm.
For example, what if we are working with a dataset that doesn't fit into
memory?

The recommended way of handling datasets is via the `tf.data` module.
Now is a good time to take some first steps with this module. Read
[the Programmer's Guide section](https://www.tensorflow.org/guide/data)
on this. You can ignore the parts on high-level APIs as well as anything
regarding TFRecords and `tf.Example` (we will get to these later) as well as
specialized topics involving time series etc. If this is still too much text for
you, [here](https://www.tensorflow.org/tutorials/load_data/numpy) is a super short
version that just covers building a dataset from numpy arrays (ignore the part
where they use Keras ;)).
For now, the main thing is that you understand how to do just that.  
Then, try to adjust your MLP code so that it uses `tf.data` to provide
minibatches instead of the class in `datasets.py`. Keep in mind that you should
 map the data into the [0,1] range (convert to float!)
and convert the labels to `int32` (check the old
 `MNISTDataset` class for possible preprocessing)!


On the E-Learning platform you
can find a little notebook (TF data Basics) that displays some basic `tf.data` stuff (also for
MNIST).

Note that the Tensorflow guide often uses the three operations `shuffle`,
`batch` and `repeat`. Think about how the results differ when you change the
order of these operations (there are six orderings in total). You can
experiment with a simple `Dataset.range` dataset. What do you think is the most
sensible order?


## First Steps with TensorBoard

As before, you will need to do some extra reading to learn how to use
TensorBoard. There are [several tutorials on the Tensorflow website](https://www.tensorflow.org/tensorboard/get_started),
accessed via
Resources -> Tools. However, they use many high-level concepts we haven't
looked at yet to build their networks, so you can find the basics
on E-Learning (Tensorboard Basics).
This is a modified version of last week's linear model that includes some lines
to do TensorBoard visualizations. It should suffice for now.
Integrate these lines into your MLP from the last assignment
to make sure you get it to work! Basic steps are just:
- Set up a file writer for some log directory.
- During training, run summary ops for anything you are interested in, e.g.
  - Usually scalars for loss and other metrics (e.g. accuracy).
  - Distributions/histograms of layer activations or weights.
  - Images that show what the data looks like.
- Run TensorBoard on the log directory.

Later, we will also see how to use TensorBoard to visualize the _computation
graph_ of a model.

Finally,
check out the [github readme](https://github.com/tensorflow/tensorboard) for
more information on how to use the TensorBoard app itself (first part of the
"Usage" section is outdated -- this is not how you create a file writer anymore).

Note: You don't need to hand in any of the above -- just make sure you get
TensorBoard to work.


## Diagnosing Problems via Visualization

Download, from E-Learning, a ZIP archive containing a few Python scripts (Deep Learning Fails).
All these contain simple MLP training scripts for MNIST. All of them
should also fail at training. For each example, find out through visualization
why this is. Also, try to propose fixes for these issues. You may want to write
summaries _every_ training step. Normally this would be too much (and slow down
your program), however it can be useful for debugging.

- These scripts/models are relatively simple -- you should be able to run them
  on your local machine as long as it's not too ancient. Of course, you will
  need to have the necessary libraries installed.
  - Otherwise, copy-pasting the scripts into a notebook cell and running it should
  also work fine.
- Please don't mess with the
parameters of the network or learning algorithm before experiencing the
original. You can of course use any oddities you notice as clues as to what
might be going wrong.
- Sometimes it can be useful to have a look at the inputs your model actually
receives. `tf.summary.image` helps here. Note that you need to reshape the
inputs from vectors to 28x28-sized images and add an extra axis for the color
channel (despite there being only one channel). Check out `tf.reshape` and
`tf.expand_dims`.
- Otherwise, it should be helpful to visualize histograms/distributions of layer
activations and see if anything jumps out. Note that histogram summaries will
crash your program in case of `nan` values appearing. In this case, see if you
can do without the histograms and use other means to find out what is going
wrong.
- You should also look at the *gradients* of the network weights; if these are "unusual"
(i.e. extremely small or large), something is probably wrong.
An overall impression of a gradient's size
can be gained via `tf.norm(g)`; feel free to add scalar summaries of these
values to TensorBoard. You can pass a `name` to the variables when defining
  them and use this to give descriptive names to your summaries.
- Some things to watch out for in the code: Are the activation functions sensible?
What about the weight initializations? Do the inputs/data look "normal"?
- Note: The final two scripts (4 and 5) may actually work somewhat, but performance
should still be significantly below what could be achieved by a "correct"
  implementation.


## What to Hand In

For each "failed" script above, provide a description of the problem as well as ideas for how
to fix it (there may be multiple ways). You can just write some text here
(markdown cells!), but feel free to reinforce your ideas with some code snippets.


## Bonus

Like last week, play around with the parameters of your networks. Use
Tensorboard to get more information about how some of your choices affect
behavior. For example, you could compare the long-term behavior of saturating
functions such as *tanh* with *relu*, how the gradients vary for different
architectures etc.

If you want to get deeper into the data processing side of things, check
[the Performance Guide](https://www.tensorflow.org/guide/data_performance).
