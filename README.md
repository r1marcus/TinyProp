# TinyProp
Training deep neural networks using backpropagation is very memory and computational intensive. This makes it difficult to train or fine-tune neural networks on embedded devices. We present an improved sparse backpropagation algorithm (TinyProp). With sparse backpropagation, only the k weights and biases of neural networks whose error gradients have the largest magnitude are trained. For this purpose, the gradients are sorted according to their magnitude and the top k gradients are selected. Normally, a constant number of k is used. TinyProp selects k adaptively. Our method decides at each training step how well the neural network is already trained and how much percent of the trainable parameters of the layer need to be calculated for the backpropagation algorithm. Our technique requires only a small calculation overhead to sort the elements of the gradient. It prunes on the basis of the magnitude of the gradient. The algorithm works particularly well on already trained networks that only need fine-tuning, which is a typical use case for embedded applications. To our knowledge, TinyProp is the first method to adaptively determine the k. For typical datasets from two datasets MNIST and DCASE2020, we have used less than 20\% of the computational budget compared to non-sparse training with an accuracy loss from 1\%. 
On average, TinyProp is 2.9 times faster than the fixed top-k and 6 \% more accurate.
