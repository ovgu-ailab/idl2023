---
layout: default
title: Assignment 6
id: ass6
---


# Assignment 6: Text Classification with RNNs (Part 2)
**Deadline: November 22nd, 9am**

Building on the last assignment, this time we want to iron out some of the issues
that were left. In particular:
- less wasteful padding,
- using embeddings to avoid one-hot vectors,
- avoiding computations on padded time steps,
- using Keras to simplify our code,
- using more advanced architectures such as LSTMs, stacked RNNs or 
  bidirectional RNNs.
  
**The notebook associated with the practical exercise can be found 
[here](https://ovgu-ailab.github.io/idl2022/assignments/6/rnns_part2.ipynb).**


## Improving training efficiency

### Within-batch padding
In the last assignment, we padded all sequences to the longest one in the dataset
because we need "rectangular" input tensors.
However, at the end of the day, only each _batch_ of inputs needs to be the same
length. If the longest sequence in a batch has length 150, the other sequences
need only be padded to that length, not the longest in the whole dataset!

Thus, if we can find a way to delay the padding after we have formed batches, we can
gain some efficiency. Unfortunately, we cannot even create a `tf.data.Dataset.from_tensor_slices`
to apply batching to!

Luckily, there are other ways to create datasets. We will be using `from_generator`,
which allows for creating datasets from elements returned by arbitrary Python
generators. Even better, there is also a `padded_batch` transformation function
which batches inputs _and_ pads them to the longest length in the batch (what
would happen if we tried the regular `batch` method?). See the notebook for a
usage example!

**Note:** Tensorflow also has `RaggedTensor`. These are special tensors allowing
different shapes per element. You can find a guide 
[here](https://www.tensorflow.org/guide/ragged_tensor). You could directly
create a dataset `from_tensor_slices` by supplying a ragged tensor, which is 
arguably easier than using a generator. Unfortunately, ragged tensors are not
supported by `padded_batch`. Sad!  
**However**, many tensorflow operations support ragged tensors, so padding can
become unnecessary in many places! You can check the guide for an example with a
Keras model. You can try this approach if you want, but for the rest of the 
assignment we will continue with the padded batches (the ragged version will likely
be very slow).

### Level 2: Bucketing
There is still a problem with the above approach. In our dataset, there are many
short sequences and few very long ones. Unfortunately, it is very likely that
all (or most) batches contain at least one rather long sequence. That means that
all the other (short) sequences have to be padded to the long one! Thus, in the
worst case, our per-batch padding might only gain us very little. It would be
great if there was a way to sort the data such that only sequences of a similar
length are grouped in a batch... Maybe there is something in the notebook?

**Note:** If you truncated sequences to a relatively small value, like 200, bucketing
may provide little benefit. The reason is that there will be so many sequences
at the exact length 200 that the majority of batches will belong to this bucket.
However, if you decide to allow a larger value, say length 500, bucketing should
become more and more effective (noticeable via shorter time spent per batch).


## Embeddings

Previously, we represented words by one-hot vectors. This is wasteful in terms 
of memory, and also the matrix products in our deep models will be very
inefficient. It turns out, multiplying a matrix with a one-hot vector simply
_extracts the corresponding column from the matrix_.

Keras offers an `Embedding` layer for an efficient implementation. Use this
instead of the one-hot operation! Note that the layer adds additional parameters,
however it can actually result in _fewer_ parameters overall if you choose a small
enough embedding size (recall the lecture discussion on using linear hidden
layers).


## RNNs in Keras

Keras offers various RNN layers. These layers take an entire 3d batch of inputs
(`batch x time x features`) and return either a complete output sequence, or only
the final output time step. There are two ways to use RNNs:
1. The more general is to define a _cell_ which implements the per-step computations,
i.e. how to compute a new state given a previous state and current input. There
  are pre-built cells for simple RNNs, GRUs and LSTMs (`LSTMCell` etc.). The cells are then put
  into the `RNN` layer which wraps them in a loop over time.
2. There also complete classes like `LSTM` which already wrap the corresponding cell.

While the first approach gives more flexibility (we could define our own cells),
it is _highly_ recommended that you stick with the second approach, as this
provides highly optimized implementations for common usage scenarios. Check the
docs for the conditions under which these implementations can be used!

Once you have an RNN layer, you can use ist just like other layers, e.g. in
a sequential model. Maybe you have an embedding layer, followed by an LSTM, 
followed by a dense layer that produces the final output. Now, you can easily
create stacked RNNs (just put multiple RNN layers one after the other), use
`Bidirectional` RNNs, etc. Also try LSTMs vs GRUs!


## Masking

One method to prevent new states being computed on padded time steps is by
using a _mask_. A mask is a binary tensor with shape `(batch x time)` with 1s
representing "real" time steps and 0s representing padding. Given such a mask,
the state computation can be "corrected" like this: 

`new_state = mask_at_t * new_state + (1 - mask_at_t) * old_state`

Where the mask is 1, the new state will be used. Where it is 0, the old state will
be propagated instead!

Masking with Keras is almost too simple: Pass the argument `mask_zero=True` to
your embedding layer (the constructor, not the call)! You can read more about
masking [here](https://www.tensorflow.org/guide/keras/masking_and_padding). The
short version is that tensors can carry a mask as an attribute, and Keras
layers can be configured to use and/or modify these masks in some way. Here,
the embedding layer "knows" to create a mask such that 0 inputs (remember that index
0 encodes padding) are masked as `False`, and the RNN layers are implemented to
perform something like the formula above.

Add masking to your model! The result should be much faster learning 
(in terms of steps needed to reach a particular performance, not time), 
in particular
with `post` padding (the only kind of padding supported by `padded_batch`). The
effect will be more dramatic the longer your sequences are.


## What to hand in

Implement the various improvements outlined in this assignment. Experiment
with adding them one by one and judge the impact (on accuracy, training time,
convenience...) of each. You can also carry out "ablation" studies where you take
the full model with all improvements, and remove them one at a time to judge their
impact. 

You can also try using higher or smaller vocabulary sizes and maximum sequence
lengths and investigate the impact of these parameters!


## Additional notes for custom RNN loops

If for some reason you are not using Keras RNN layers, but rather your own loops
over time, there are a few more things to be aware of when using `tf.function`:
1. There seems to be an issue related to data shapes when using `bucket_by_sequence_length`
and the final batch in the dataset (which can be smaller than the others). If you
   receive strange errors about unknown data shapes, you can set `drop_remainder=True`,
   or use regular `padded_batch` instead of bucketing.
2. A `tf.function` is re-compiled every time it receives an input with a different
"signature". This is defined as the shape and data type of the tensor. When every
   batch has a different sequence length, this causes the training loop to be
   re-compiled every step. You can fix this by supplying an `input_signature` to
   `tf.function` -- please check the API docs. You can also pass `experimental_relax_shapes=True`
   instead, although this seems to be a little less effective.
   
