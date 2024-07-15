This is a little experiment with the goal of understanding more about machine learning.

It is based on 3Blue1Brown's series on basic deep learning:
https://www.youtube.com/watch?v=aircAruvnKk

And the simple network code from:
https://github.com/mnielsen/neural-networks-and-deep-learning/blob/master/src/network.py

Here I implemented a very simple neural network in 100% Odin that runs on the CPU.

It compiles into a small exe that reads a 28x28 `digit.png` and attempts to predict what digit it is.

It is trained off of the mnist dataset, and you can train it yourself by running the `train` function in main.

If you don't have `model.txt` in the folder, it won't let you compile since I bake it into the exe, so you will need to comment out the `predict_digit` function definition if you want to train and you don't have the model file.

I didn't make any attempt to process the training data to make it more robust so the numbers have to be in the middle of the image for it to work well, and even then it can be a bit wonky with its predictions on some digits.