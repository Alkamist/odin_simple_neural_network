This is a little experiment with the goal of understanding more about machine learning.

It is based on 3Blue1Brown's series on basic deep learning:
https://www.youtube.com/watch?v=aircAruvnKk

And the simple network code from:
https://github.com/mnielsen/neural-networks-and-deep-learning/blob/master/src/network.py

It is also based on the article found here:
https://gpuopen.com/learn/deep_learning_crash_course/

Here I implemented a simple multi-layer perceptron in 100% Odin that runs on the CPU.

There are 3 main functions that you can run:
- train
- load_and_validate
- predict_digit

`train` will train the model off of the MNIST dataset, which you will need to unzip first. It will dump the model's best performing weights and biases into a json file as it is training. You should probably run it with `-o:speed`.

`load_and_validate` will load the model from the json file and then run it on the MNIST validation dataset to check for accuracy.

`predict_digit` will load the model from the json file and try to predict what digit is present in the file `digit.png`.

The model uses:
- Leaky Relu
- Cross Entropy Loss
- Dropout
- He and Xavier Initialization
- L2 Regularization
- ADAM Optimization

And a quick note: All of the digits in the MNIST dataset are centered, I haven't yet made a way to augment the dataset to be more robust to off-center and oddly scaled digits, so if you draw a digit in `digit.png` make sure it is in the center.