---
layout: default
title: Assignment 8
id: ass8
---

# Assignment 8: Word2Vec
**Deadline: December 6th, 9am**


In this week, we will look at "the" classic model for learning word embeddings.
This will be another tutorial-based assignment.
[Find the link here](https://www.tensorflow.org/tutorials/text/word2vec).

The key points are:
- Getting to know an example of _self-supervised learning_, where we have data
without labels, and are constructing a task directly from the data (often some
  kind of prediction task) in order to learn deep representations,
- Understanding how Softmax with a very large number of classes is problematic,
and getting to know possible workarounds,
- Exploring the idea of word embeddings.   


## Questions for Understanding

As in the last assignment, answer these questions in your submission to make sure
you understand what is happening in the tutorial code!

1. Given the sentence "I like to cuddle dogs", how many skipgrams are created with
  a window size of 2? 
2. In general, how does the number of skipgrams relate to the
  size of the dataset (in terms of input-target pairs)?
3. Why is it not a good idea to compute the  full softmax for classification?  
4. The way the dataset is created, for a given `(target, context)` pair, are the
  negative samples (remember, these are randomly sampled) the same each time this
  training example is seen, or are they different?
5. For the given example dataset (Shakespeare), would the code create `(target, context)`
  pairs for sentences that span multiple lines? For example, the last word of one
  line and the first word of the next line?
6. Does the code generate skipgrams for padding characters (index 0)?  
7. The `skipgrams` function uses a "sampling table". In the code, this is shown
to be a simple list of probabilities, and it is created without any reference to
  the actual text data. How/why does this work? I.e. how does the program "know"
  which words to sample with which probability?
  

## Possible Improvements & Extensions

- _If_ the code generates skipgrams for padding characters: This is probably not
a good idea. Can you prevent this from happening?
- _If_ the code is not re-drawing negative samples each iteration: Can you change it
so that it does? This may give less biased results.
- The candidate sampler may accidentally draw the true context word as one
  of the negative words. Can you find a way to detect and avoid this? Note that
  there is `tf.nn.sampled_softmax_loss` which supports such an argument. Using
  this would require significant re-writing of the code, however (e.g. getting
  rid of the `uniform_candidate_sampler` entirely).
- One of the most "impressive" features of these word embeddings is that, given
a well-trained model, analogies can be performed via vector arithmetic. Try this:
  - Get the learned vectors (either target or context embeddings) for the words
    `king, queen, man, woman`. Of course, this assumes that these words are present
    in the training data.
  - Compute the vector `king - man + woman`.
  - Compute the similarity between the resulting vector and _all_ word vectors.
    Which one gives the highest similarity? It "should" be `queen`. Note that it
    might actually be `king`, in which case `queen` should at least be second-highest.
    To compute the similarity, you should use cosine similarity.
  - You can try this for other pairs, such as `Paris - France + Germany = Berlin` etc.
- Use a larger vocabulary and/or larger text corpora to train the models. See 
how embedding quality and training effort changes. You can also implement a version
  using the "naive" full softmax, and see how the negative sampling increases in
  efficiency compared to the full version as the vocabulary becomes larger!


## Optional: CBOW Model

The tutorial only covers the Skipgram model, however the same paper also proposed
the (perhaps more intuitive) _Continuous Bag-of-Words_ model. Here instead of
predicting the context from the center word, it's the other way around. If you
are looking for more of a challenge implementing a model by yourself, the changes
should be as follows:
- In CBOW, each training example consists of _multiple_ context words and a single
target word. There is no equivalent to the `skipgrams` preprocessing function,
  but you can simply iterate over the full text data in small windows (there is
  `tf.data.Dataset.window` which may be helpful here) and for each window use
  the center word as the target and the rest as context.
- The context embedding is computed by embedding all context words separately,
and then averaging their embeddings.

The rest stays pretty much the same. You will still need to generate negative
examples through sampling, since the full softmax is just as inefficient as with
the Skipgram model.

Compare the results of the CBOW model with the Skipgram one!
