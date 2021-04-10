# Generative Adversarial Network (GAN)
GANs is a relatively new technique to a sublass of machine learning problems known as *"generative modelling"* employed with the help of **deep learning methods**.\
The idea of such a model is to identify a certain set of *regularities* in a large batch of data, and use this to train a model which can "generate" *plausible* new examples that seem as though they are from the same dataset.\
In its essence, there are **2** sub models to this problem: A **generator**(for generating new examples) and a **discriminator**(for testing the plausibility of the generated samples).

# Overview
At a very abstract level, I have trained a neural network to identify digits from 0-9 in the **Devanagari script** and piped this predicted output to the GAN which then generates examples from the (English) **MNIST dataset**.\
Note that since I have made a very basic implementation of the GAN (ie-without the use of any *major* scientfic libraries like ```Tensorflow```, ```PyTorch``` etc.), I feed in a single (*homogenous*) digit list at a time to the GAN for generating examples. This slow process is largely attributed to the fact that the aforementioned scientific libraries run computations primarily on the GPU as opposed to calculations on the CPU (done in the case of ```numpy```).

## Devanagari Digit classifier
I was lucky enough to find the CSV files of the MNIST images of the devanagari script (albeit after a lot of googling :P) to save myself some time. The ```zip``` file can be found on [Kaggle](https://www.kaggle.com/ashokpant/devanagari-character-dataset-large), and the CSV format of the same is found [here](https://github.com/sknepal/DHDD_CSV).\
I used the ```MLPClassifier``` function from the ```scikit``` module to train the neural network. After a bit of trial and error, working with **80 hidden layers**, and an initial learning rate of 0.1 and a ```max_epoch``` value of 100, a fairly high prediction accuracy was achieved.\
Although the ```max_epoch``` value is set at 100, it seldom takes those many steps, and converges a fair amount of iterations before.
### Result
The neural net converges fairly quickly as can be seen below
![Epoch and cost value](https://github.com/ckapoor7/bare-bones-GANs/blob/main/Screen%20Shot%202021-04-10%20at%2010.11.17%20PM.png)
A fairly high accuracy of **98.75%** was achieved with the help of this model. 

## Vanilla GAN
### Result


## The final thing
### Result

# Drawbacks

# References
The primary resource I used for my implementation for this was Ian Goodfellow's research paper available [here](https://arxiv.org/pdf/1406.2661.pdf).\
In some places (**cross-entropy loss calculation**) I have changed my implementation with respect to what is described in the paper to make it computationally less expensive whilst achieveing a nearly similar result if I would have done otherwise.


Given the Devnagri MNIST Dataset and the regular MNIST Dataset train a model to generate regular MNIST numbers given a Devnagri Number as input.
