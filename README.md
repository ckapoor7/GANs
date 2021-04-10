# Generative Adversarial Network (GAN)
GANs is a relatively new technique to a sublass of machine learning problems known as *"generative modelling"* employed with the help of **deep learning methods**.\
The idea of such a model is to identify a certain set of *regularities* in a large batch of data, and use this to train a model which can "generate" *plausible* new examples that seem as though they are from the same dataset.\
In its essence, there are **2** sub models to this problem: A **generator**(for generating new examples) and a **discriminator**(for testing the plausibility of the generated samples). 

#Installation

##Vanilla GAN
###Result


##Generator and discriminator
###Result

##The final thing
###Result

#Drawbacks

#References
The primary resource I used for my implementation for this was Ian Goodfellow's research paper available [here](https://arxiv.org/pdf/1406.2661.pdf).\
In some places (**cross-entropy loss calculation**) I have changed my implementation with respect to what is described in the paper to make it computationally less expensive whilst achieveing a nearly similar result if I would have done otherwise.


Given the Devnagri MNIST Dataset and the regular MNIST Dataset train a model to generate regular MNIST numbers given a Devnagri Number as input.
