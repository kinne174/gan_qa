# Generative Adversarial Network for Question Answering

The goal of this project is adversarially attack a Classifier with unseen data to
make it more robust to false information. To do this a Generative Adversarial
Network (GAN) is used to fabricate contexts linking the question and answer that
will ultimately make the Classifier select wrong. Elements of Reinforcement Learning
are used to train the Generator while the Discriminator and Classifier are trained
using standard Cross Entropy loss.

The Generator is provided with the words for the (question, answer, context) 
and replaces words of the context to strengthen the Classifier against faulty
information. The Discriminator is used in the standard way, to ensure samples
from the Generator are reflective of the true distribution of words seen given
the surrounding sentence context. 

The models used are Google's Transformers<sup>1</sup> from the huggingface<sup>2</sup> 
repository for the Generator and Classifier and a standard LSTM RNN for the Discriminator.

A visual of the network is provided here:
![network visual](https://github.com/kinne174/gan_qa/)