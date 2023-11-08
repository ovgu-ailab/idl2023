---
layout: default
title: Assignment 5
id: ass5
---


# Assignment 5: Text Classification with RNNs (Part 1)
**Deadline: November 15th, 9am**

In this assignment and the next, we are switching to a different modality of data: Text.
Namely, we will see how to assign a single label to input sequences of arbitrary length.
This has many applications, such as detecting hate speech on social media or
detecting spam emails.
Here, we will look at sentiment analysis, which is supposed to tell what kind of
emotion is associated with a piece of text.

In part 1, we are mainly concerned with implementing RNNs at the low level so that
we understand how they work in detail.
The models themselves will be rather rudimentary.
We will also see the kinds of problems that arise when working with sequence data,
specifically text.
Next week, we will build better models and deal with some of these issues.

**The notebook associated with the practical exercise can be found 
on E-Learning (TODO UPLOAD)**


## The Data

We will be using the IMDB movie review dataset. This dataset comes with Keras
and consists of 50,000 movie reviews with binary labels (positive or negative),
divided into training and testing sets of 25,000 sequences each.

### A first look
The data can be loaded the same way as MNIST or CIFAR  --
`tf.keras.datasets.imdb.load_data()`.
If you print the sequences, however, you will see that they are numbers, not text.
Recall that deep learning is essentially
[a pile of linear algebra](https://xkcd.com/1838/).
As such, neural networks cannot take text as input, which is why it needs to be
converted to numbers.
This has already been done for us -- each word has been  replaced by a number,
and thus a movie review is a sequence of numbers (punctuation has been removed).

If you want to restore the text, `tf.keras.datasets.imdb.get_word_index()` has
the mapping -- see the notebook for how you can use this, as well as some
additional steps you need to actually get correct outputs.

### Representing words
Our sequences are numbers, so they can be put into a neural network. But does
this make sense?
Recall the kind of transformations a layer implements: A linear map followed by a
(optional) non-linearity.
But that would mean, for example, that the word represented by index 10 would be
"10 times as much" as the word represented by index 1. And if we simply swapped
the mapping (which we can do, as it is completely arbitrary), the roles would be
reversed! Clearly, this does not make sense.

A simple fix is to use one-hot vectors: Replace a word index by a vector with as
many entries as there are words in the vocabulary, where all entries are 0 except
the one corresponding to the respective word, which is 1 -- see the notebook.

Thus, each word gets its own "feature dimension" and can be transformed separately.
With this transformation, our data points are now sequences of one-hot vectors,
with shape `(sequence_length, vocabulary_size)`.

### Variable sequence lengths
Of course, not all movie reviews have the same length.
This actually represents a huge problem for us:
We would like to process inputs in batches, but tensors generally have to be
"rectangular", i.e. we cannot have different sequence lengths in the same batch!
The standard way to deal with this is _padding_: Appending additional elements
to shorter sequences such that all sequences have the same length.

In the notebook, this is done in a rather crude way:
All sequences are padded to the length of the longest sequence in the dataset.

**Food for thought #1:** Why is this wasteful? Can you think of a smarter padding
scheme that is more efficient? Consider the fact that RNNs can work on arbitrary
sequence lengths, and that training minibatches are pretty much independent of each
other.

### Dealing with extremes
Once we define the model, we will run into two issues with our data:
1. Some sequences are very long. This increases our computation time as well as
   massively hampering gradient flow. It is highly recommended that you limit the
   sequence length (200 could be a good start). You have two choices:
   1. _Truncate_ sequences by cutting off all words beyond a limit. Both `load_data`
    and `pad_sequences` have arguments to do this. We recommend the latter as you
      can choose between "pre" or "post" truncation.
   2. _Remove_ all sequences that are longer than a limit from the dataset. Radical!
2. Our vocabulary is large, more than 85,000 words. Many of these are rare words
which only appear a few times. There are two reasons why this is problematic:
   1. The one-hot vectors are huge, slowing down the program and eating memory.
   2. It's difficult for the network to learn useful features for the rare words.

   `load_data` has an argument to keep only the `n`
   most common words and replace less frequent ones by a special "unknown word"
   token (index 2 by default). As a start, try keeping only the 20,000 most common words or so.

**Food for thought #2:** Between truncating long sequences and removing them, which
option do you think is better? Why?

**Food for thought #3:** Can you think of a way to avoid the one-hot vectors
completely? Even if you cannot implement it, a conceptual idea is fine.

With these issues taken care of, we should be ready to build an RNN!


## Building The Model

A Tensorflow RNN "layer" can be
confusing due to its black box character: All computations over a full sequence
of inputs are done internally. **To make sure you understand how an RNN "works",
you are asked to implement one from the ground up, defining variables yourself
and using basic operations such as `tf.matmul` to define the computations at
each time step and over a full input sequence.** There are some related tutorials
available on the TF website, but all of these use Keras.

For this
assignment, you are asked **not** to use the `RNNCell` classes nor any related Keras
functionality. Instead, you should study the basic RNN equations and "just"
translate these into code. You can still use Keras optimizers, losses etc.
**You can also use `Dense` layers instead of low-level ops,
but make sure you know what you are doing.**
You might want to proceed as follows:

- On a high level, **nothing about the training loop changes!** The RNN gets an input
and computes an output. The loss is computed based on the difference between
  outputs and targets, and gradients are computed and applied to the RNN weights,
  with the loss being backpropagated _trough time_.
- The differences come in how the RNN computes its output. The basic recurrency
can be seen in equation 10.5 of the deep learning book, with more details in
  equations 10.8-10.11. The important idea is that, at each time step, the RNN
  essentially works like an MLP with a single hidden layer, but two inputs
  (last state and current input). In total, you need to "just":
    - Loop over the input, at each time step taking the respective slice. Your
    per-step input should be `batch x features` just like with an MLP!
    - At each time step, compute the new state based on the previous state as well
    as the current input.
    - Compute the per-step output based on the new state.
- What about comparing outputs to targets? Our targets are simple binary labels.
On the other hand, we have one output _per time step_. The usual approach is to
  discard all outputs except the one for the very last step. Thus, this is a
  "many-to-one" RNN (compare figure 10.5 in the book).
- For the output and loss, you actually have two options:
  1. You could have an output layer with 2 units, and use sparse categorical
    cross-entropy as before (i.e. softmax activation). Here, whichever output is
     higher "wins".
  2. You can have _a single_ output unit and use binary cross-entropy (i.e.
     sigmoid activation). Here, the output is usually thresholded at 0.5.

**Food for thought #4:** How can it be that we can _choose_ how many outputs we
have, i.e. how can both be correct? Are there differences between both choices
as well as (dis)advantages relative to each other?

### Open Problems
#### Initial state
To compute the state at the first time step, you would need a "previous state",
but there is none. To fix this, you can define an "initial state" for the network.
A common solution is to simply use a tensor filled with zeros. You could also add
a trainable variable and learn an initial state instead!

**Food for thought #5:** All sequences start with the same special "beginning of
sequence" token (coded by index 1). Given this fact, is there a point in learning
an initial state? Why (not)?

#### Computations on padded time steps
Recall that we padded all sequences to be the same length. Unfortunately, the RNN
is not aware that we did this. This can be an issue, as we are basically computing
new states (thus computing outputs as well as influencing future states) based
on "garbage" inputs.

**Food for thought #6:** `pad_sequences` allows for `pre` or `post` padding. Try
both to see the difference. Which option do you think is better? Recall that
we use the final time step output from our model.

**Food for thought #7:** Can you think of a way to prevent the RNN from computing
new states on padded time steps? One idea might be to "pass through" the
previous state in case the current time step is padding. Note that, within a batch,
some sequences might be padded for a given time step while others are not.

#### Slow learning
Be aware that it might take several thousand steps for the loss to start moving
at all, so don't stop training too early if nothing is happening.
Experiment with weight initializations and learning rates. For fast learning,
the goal is usually to set them as large as possible without the model "exploding".

A major issue with our "last output summarizes the sequence" approach is that the
information from the end has to backpropagate all the way to the early time steps,
which leads to extreme vanishing gradient issues. You could try to use the RNN
output more effectively. Here are some ideas:

- Instead of only using the final output, average (or sum?) the logits (pre-sigmoid) of all
time steps and use this as the output instead.
- Instead of the logits, average the _states_ at all time steps and compute the
output based on this average state. Is this different from the above option?
- Compute logits and sigmoids for each output, and average the per-step probabilities.

**Food for thought #8:** What could be the advantage of using methods like the above?
What are disadvantages? Can you think of other methods to incorporate the full
output sequence instead of just the final step?

## What to hand in

- A low-level RNN implementation for sentiment classification. If you can get it
to move away from 50% accuracy on the training set, that's a success. Be wary of
  overfitting, however, as this doesn't mean that the model is generalizing! If
  the test (or validation) loss isn't moving, try using a smaller network. Also
  note that you may sometimes get a higher test accuracy, while the test loss
  is _also_ increasing (how can this be?)!
- Consider the various questions posed throughout the assignment and try to answer
them! You can use text cells to leave short answers in your notebook.
