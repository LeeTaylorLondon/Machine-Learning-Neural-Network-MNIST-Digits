# ML Neural Network MNIST Handwritten Digits
## Description 
This project features a sequential neural network model trained to recognise hand-written digits.
There are two files, train_model.py which contains the functions and structure of the model. Secondly, there is
interactive.py which provides a small GUI screen to draw a digit 0-9. The user's drawing can be inputted into the 
trained model by pressing a key in which the neural network outputs its guess.

## Installation
* Pip install h5py (built with 3.1.0)
* Pip install tensorflow (built with 2.5.0)
* Pip install numpy (built with 1.19.5)
* Pip install pygame (built with 2.0.1 - for interactive.py)
* Download below files, create folder /mnist_data, and place downloaded files in created folder  
  http://www.pjreddie.com/media/files/mnist_train.csv  
  http://www.pjreddie.com/media/files/mnist_test.csv

## Usage
Before running any python files please make sure you have completed ALL above installation steps.

Read train_model.py function build_model, to see the structure of the neural network.  
If you run train_model.py, it will build and train a new model which will overwrite the existing saved model
and weights.
The newly trained model will be evaluated, and it's accuracy outputted.    

![Image of interactive.py](/images/Capture.PNG)

Upon running interactive.py, a small window will pop up. You are able to draw in the bordered off box. 
Above the drawing section there are two key prompts, pressing C, deletes everything you've drawn on screen. 
Pressing T queries the trained model, a text along the bottom of the border displays the neural network's 
guess. 

## Neural Network Details
Input layer 784 units, activation function sigmoid.  
Input vector is divided by 255.0 to reduce input to values between 0 and 1.  
Hidden layer 500 units, activation function sigmoid.  
Output layer 10 units, activation function sigmoid.  
Optimizer is Adam, loss is mean squared error.  
  
Testing accuracy: ~98.14%

## Credits
* Author: Lee Taylor

## Note
This is my first interative machine learning project, inspired by the book,
Make Your Own Neural Network by Tariq Rashid. 
