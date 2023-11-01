---
layout: default
title: Assignment 4
id: ass4
---


# Assignment 4: Graphs & ResNets
**Deadline: November 8th, 9am**


## Graph-based Execution

**Note: This section is not necessary for submission, but it is highly recommended
that you read it, as it will significantly speed up your models.**

So far, we have been using so-called "eager execution" exclusively: Commands are
run as they are defined, i.e. writing `y = tf.matmul(X, w)` actually executes
the matrix multiplication.

In Tensorflow 1.x, things used to be different: Lines like the above would only
_define the computation graph_ but not do any actual computation. This would be
done later in dedicated "sessions" that execute the graph. Later, eager 
execution was added as an alternative way of writing programs and is now the
default, mainly because it is much more intuitive/allows for a more natural
workflow when designing/testing models.

Graph execution has one big advantage: It is very efficient because entire
models (or even training loops) can be executed in low-level C/CUDA code without
ever going "back up" to Python (which is slow). As such, TF 2.0 still retains
the possibility to run stuff in graph mode if you so wish -- let's have a look!

As expected, there is [a tutorial 
on the TF website](https://www.tensorflow.org/guide/intro_to_graphs), as well as 
[this one](https://www.tensorflow.org/guide/function)
which goes intro extreme depth on all the subtleties. The basic gist is:
- You can annotate a Python function with `@tf.function` to "activate" graph
execution for this function.
- The first time this function is called, it will be _traced_ and converted to
a graph.
- Any other time this function is called, _the Python function will not be run;
instead the traced graph is executed_.
- The above is not entirely true -- functions may be _retraced_ under certain
(important) conditions, e.g. for every new "input signature". This is treated in
detail in the article linked above.
- Beware of using Python statements like `print`, these will not be traced so
the statement will only be called during the tracing run itself. If you want to
print things like tensor values, use `tf.print` instead. Basically, traced TF
functions only do "tensor stuff", not general "Python stuff".

Go back to some of your pevious models and sprinkle some `tf.function` annotations
in there. You might need to refactor slightly -- you need to actually wrap things
into a function!
- The most straightforward target for decoration is a "training step" function
that takes a batch of inputs and labels, runs the model, computes the loss and
the gradients and applies them.
- In theory, you could wrap a whole training loop (including iteration over a
dataset) with a `tf.function`. If you can get this to work on one of your
previous models _and actually get a speedup_, you get a cookie. :)
  - That is to say,
  expensive iterations etc. perhaps should not be wrapped in a graph. 
  - Also, you often
  want to do "Python stuff" (keeping track of metrics, doing plots, printing to console
  etc.) in the training loop, so wrapping the entire loop is usually not feasible anyway.

**Note:** In recent TF versions, another speedup factor was added: You can specify
so-called "just-in-time (JIT) compilation" for your graphs. Simply use `@tf.function(jit_compile=True)`.
This will increase compilation time, but further reduce execution time as well as memory usage!
However, there are some edge cases where JIT compilation doesn't work, so if you ever
run into strange errors, try turning it off.


## ResNet

Previously, we saw how to build neural networks in a purely sequential manner --
each layer receives one input and produces one output that serves as input to
the next layer. There are many architectures that do not follow this simple
scheme. You might ask yourself how this can be done in Keras. One answer is via
the so-called functional API. There is an in-depth guide 
[here](https://www.tensorflow.org/guide/keras/functional). Reading just the intro
should be enough for a basic grasp on how to use it, but of course you can read
more if you wish.

Next, use the functional API to implement a 
[ResNet](https://arxiv.org/abs/1512.03385). This is an incredibly important architecture;
residual connections are part of pretty much every state-of-the-art model in any
domain.

You do _not_ need to follow the exact same architecture from the paper, in fact you will probably
want to make it smaller for efficiency reasons. Just make sure you have one or
more "residual blocks" with multiple layers each. 
You can also leave out batch
normalization (this will be treated later in the class) as well as "bottleneck
layers" (1x1 convolutions) if you want. Still, it can be a good exercise to see
how best to structure your code to easily build more complex and arbitrarily deep
models. Let's say, how would you build a network with 20-100 layers with as little
code duplication as possible?

**Bonus:** Can you implement ResNet with the Sequential API? You might want to look
at how to 
[implement custom layers](https://www.tensorflow.org/guide/keras/custom_layers_and_models)
(shorter version 
[here](https://www.tensorflow.org/tutorials/customization/custom_layers))...


## What to Hand In

- ResNet. Thoroughly experiment with (hyper)parameters. Try to achieve the best
performance you can on CIFAR10/100.
- For your model(s), compare performance with and without `tf.function`. You can
  also do this for non-ResNet models. How does the impact depend on the size
  of the models?

The part is just here for completeness/reference, to show some additional TensorBoard functionalities. Check
it out if you want.


## Bonus Reading: TensorBoard Computation Graphs

You can display the computation graphs Tensorflow uses internally in TensorBoard.
This can be useful for debugging purposes as well as to get an impression what
is going on "under the hood" in your models. More importantly, this can be combined
with _profiling_ that lets you see how much time/memory specific parts of your
model take.

To look at computation graphs, you need to _trace_ computations explicitly.
See the last part of [this guide](https://www.tensorflow.org/tensorboard/graphs)
for how to trace `tf.function`-annotated computations. Note: It seems like you
have to do the trace the first time the function is called (e.g. on the first
training step).
