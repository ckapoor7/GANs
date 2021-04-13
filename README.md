# Generative Adversarial Network (GAN)
GANs is a relatively new technique to a subclass of machine learning problems known as *"generative modelling"* employed with the help of **deep learning methods**.\
The idea of such a model is to identify a certain set of *regularities* in a large batch of data, and use this to train a model which can "generate" *plausible* new examples that seem as though they are from the same dataset.\
In its essence, there are **2** sub models to this problem: A **generator**(for generating new examples) and a **discriminator**(for testing the plausibility of the generated samples).

# Overview
At a very abstract level, I have trained a neural network to identify digits from 0-9 in the **Devanagari script** and piped this predicted output to the GAN which then generates examples from the (English) **MNIST dataset**.\
Note that since I have made a very basic implementation of the GAN (ie-without the use of any *major* scientfic libraries like ```Tensorflow```, ```PyTorch``` etc.), I feed in a single (*homogenous*) digit list at a time to the GAN for generating examples. This slow process is largely attributed to the fact that the aforementioned scientific libraries run computations primarily on the GPU as opposed to calculations on the CPU (done in the case of ```numpy```).

## Devanagari Digit classifier
I was lucky enough to find the CSV files of the MNIST images of the devanagari script (albeit after a lot of googling :P) to save myself some time. The ```zip``` file can be found on [Kaggle](https://www.kaggle.com/ashokpant/devanagari-character-dataset-large), and the CSV format of the same is found [here](https://github.com/sknepal/DHDD_CSV).\
I used the ```MLPClassifier``` function from the ```scikit``` module to train the neural network. After a bit of trial and error, working with **80 hidden layers**, and an initial learning rate of 0.1 and a ```max_epoch``` value of 100, a fairly high prediction accuracy was achieved.\
Although the ```max_epoch``` value is set at 100, it seldom takes those many steps, and converges a fair amount of iterations before. [This](https://colab.research.google.com/drive/1HkDEJfKoFRFzJh6OMn1LGIxiejoVsuKS#scrollTo=FQIysSno7Nax) is the link to the Google Colab notebook.
### Result
The neural net converges fairly quickly as can be seen below
![Epoch and cost value](https://github.com/ckapoor7/bare-bones-GANs/blob/main/results/nn-loss.png)
A fairly high accuracy of **98.75%** was achieved with the help of this model. 
![Model accuracy](https://github.com/ckapoor7/bare-bones-GANs/blob/main/results/nn-accuracy.png)
I have also generated a table for the **actual Vs predicted** values of the digits from the validation set
![validation set](https://github.com/ckapoor7/bare-bones-GANs/blob/main/results/nn-pred.png)

## Vanilla GAN
A *"vanilla GAN"* for the sole reason that it is a very rudimentary implementation, which is **not at all** optimized for speed. This is a complete end to end implmentation from scratch done in the large part using ```numpy``` arrays.  As an input, I pass it a list of a **single digit** from 0-9 which consequently starts the generation process of fake examples. A total of **100** ```epochs``` seems to be sufficient in producing plausible images.\
I map the examples onto a 4x4 grid. Since the input pixels were scaled to a value between 0-1, for generating the I have converted the scaled feature back to grayscale values. Also, for the ease of loading in the CSV files for the MNIST digits, I used the ```keras``` backend framework to cut down (a tad bit significantly) on my work. At an average, training the GAN for a single digit seems to take about **3 minutes** which is not too bad for a first attempt:)\
The Google colab notebook can be found [here](https://colab.research.google.com/drive/1P7bhxQaUWDE-b3WcbfoIdMmB_ovtVOxZ)
### Result
I ran the code for generating examples of the digit 0.\
Starting out at ```epoch``` 0, we see how the GAN begins its learning procedure...\
![epoch 0](https://github.com/ckapoor7/bare-bones-GANs/blob/main/results/vg-itr0.png)
...and we have something much more plausible towards the end
![epoch 95](https://github.com/ckapoor7/bare-bones-GANs/blob/main/results/vg-itr95.png)
Note that the **learning rate (LR)** decays steadily in accordance with our **decay rate** (10^-4).


## The final thing
The final thing is an amalgamation of the Vanilla GAN model as well as the digit classifier. I take a Devanagari digit image as an input and generate an MNIST digit in english as the output. Concretely, the ouput of the digit classifier can be thought of as being *"piped"* into the GAN model. Everything else remains the same as described above.\
The notebook for this can be found [here](https://colab.research.google.com/drive/1YxvCLGCMGvwXd5LM4NTGwWcb56pMyn0U#scrollTo=S_d8m1XBJcBM)
### Result
I ran the code for a known digit from the Devanagari set (from the table). In the input that I have provided to the generator, ```y_pred[3]``` corresponds to digit 5.\
For ```epoch``` 0, we have a bunch of seemingly random pixels
![epoch 0](https://github.com/ckapoor7/bare-bones-GANs/blob/main/results/final-itr0.png)
And towards the end, we have something better than what we started with (that passes as an image of digit 5)
![epoch 95](https://github.com/ckapoor7/bare-bones-GANs/blob/main/results/final-itr95.png)
This took 2 minutes and 54 seconds to complete execution.

# References
The primary resource I used for my implementation for this was Ian Goodfellow's research paper available [here](https://arxiv.org/pdf/1406.2661.pdf). Of course since I am not really well versed with some terms in the paper, a ton of googling did serve its purpose:P\
In some places (**cross-entropy loss calculation**) I have changed my implementation with respect to what is described in the paper to make it computationally less expensive whilst achieveing a nearly similar result if I would have done otherwise.\
I have also used a technique for initializing the weight matrix which was not known to me before known as the **Xavier initialization**. [This](https://www.deeplearning.ai/ai-notes/initialization/) article gives a really nice overview and mathematical reasoning behind its efficacy.


