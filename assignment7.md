---
layout: default
title: Assignment 7
id: ass7
---

# Assignment 7: Attention-based Neural Machine Translation
**Deadline: November 29th, 9am**

**Note: This assignment was changed from last year, moving from an encoder-decoder
with attention to a Transformer architecture. It's the first time we try it and
a bit experimental.**

In this task, you will implement a model for neural machine translation using
a Transformer.
We will follow [a TF tutorial](https://www.tensorflow.org/text/tutorials/transformer)
for this purpose.

**Do not just run the code and call it a day. To make sure you understand what is going
on, you need to answer some questions, posted further below!**


## Notes on the Tutorial

**READ THIS**

- You can copy and paste the code from the tutorial into your own notebook, **or**
use the "Run in Google Colab" at the top.
  - If you go for option 2, make sure you "Save a copy in Drive" in the File menu,
  **else your notebook will not be saved**. Also, after saving a copy, **remove the tick**
  for "Omit cell outputs when saving this notebook" in the settings!
  - Also, please **do not submit the full notebook with all the architecture images**.
  This just bloats the submission unnecessarily! Remove the cells with images before
  submission!
- **Do not** run the code cell with `pip install` in the Setup section! This might
mess up your environment! The **only**
library you need to install is `!pip install tensorflow-text==2.14.0`!
  - If you don't install the correct version of tf-text, it will install TF 2.15 instead,
  which for whatever reason is MUCH slower (were talking 30x or so). Training will
  take forever!


## Questions for Understanding

Here are a few questions for you to check how well you understood the tutorial.  
**Please answer them (briefly) in your solution!**

1. Which parts of the sentence are used as tokens? Characters? Word? 
Or something else (if so, what)?
2. Do the same tokens in different language have the same ID? For example, 
would the same token index map to the German word `die` and to the 
  English word `die`?
3. Why does the Transformer require positional encodings? Optional: What do you
think is the point of using sine waves of different frequencies?
3. Yes or No: At each position, the decoder is attending to all previous positions, 
 i.e. all encoder positions and the previous decoder predictions.
4. The decoder uses teacher forcing. Does this mean the time steps can be computed
 in parallel?
5. Can the encoder time steps be computed in parallel?
6. Why is a mask applied to the loss function?
7. Think about the loss function and metrics used (cross-entropy & accuracy). What
exactly do these measure? Do you think this is a good measure of translation quality?
If not, why not?
7. When translating the same sentence multiple times, do you get the same result?
Why (not)? If not, what changes need to be made to get the same result each time?
8. Which components would we need to switch out/change if we wanted to use a different pair
of languages?

Hand in all of your code, i.e. the working tutorial code along with all changes/additions you made. Include outputs which document some of your experiments. Also
remember to answer the questions above! Of course you can also write about other
observations you made.
