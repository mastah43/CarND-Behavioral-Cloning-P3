#**Behavioral Cloning** 

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* video.mp4 a video where the trained model drives the car on track 1
* writeup_report.md or writeup_report.pdf summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track 1 
by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. 
The file shows the pipeline I used for training and validating the model, 
and it contains comments to explain how the code works.

The following parameters can be used:
* batch_size: batch size to use for training and validation
* epochs: number of epochs to train
* limit_samples: optional; can be set to use only the given number of samples from the data sets
* plot_augmented_images: optional; can be set to plot the first 30 images used for training

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

I implemented the model described in a paper by nvidia for end 2 end vehicle steering.
It is a regression model which outputs the steering angle in a range of -1.0 to +1.0 which interpretes to -25 to +25 
degrees. It solely takes as input the RGB image of the simulator with a width of 320 pixels and a height of 160 pixels.
The model consists of a cropping layer, a normalization layer, five convolutional layers and five fully connected layers.


TODO add link to paper

TODO
The model includes RELU layers to introduce nonlinearity (code line 20), and the data is normalized in the model using a Keras lambda layer (code line 18). 

####2. Attempts to reduce overfitting in the model

To prevent overfitting I included dropout layers in the model after each of the first three fully connected layers.
The model was trained and validated on different data sets (20% for validation) to ensure that the model was not overfitting. 

####3. Model parameter tuning

The model training uses an adam optimizer, so the learning rate was not tuned manually.
I did not spend time on tuning the model parameters like dropout factor or layer dimensions.
Instead I experimented with different amounts of training data and augmentation techniques.

####4. Appropriate training data

As a training data set I used to provided data set by Udacity.
Additionally I collected several rounds of clean driving on track 1 to include more variance.

Additionally to the center images and steering angles I also used the left and right camera image and  
a steering angle +6 respectively -6 degrees in relation to the center steering angle as training input.
This helped to make the model learn recovery on all parts of the track. 

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

I did not use a model pre
TODO
The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

TODO
The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

####3. Creation of the Training Set & Training Process

TODO
To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

TODO
I did not include explicit recordings for recovering from sides to the center of the road. 
Instead I used the left and right camera images from all recordings. 
This approach should get a generalized model that is able to recover from much more side driving situations.

![alt text][image3] TODO image from left camera
![alt text][image4] TODO image from left camera
![alt text][image5]

TODO
To augment the data set, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

TODO
After the collection process, I had X number of data points. I then preprocessed this data by ...

TODO
I finally randomly shuffled the data set and put 20% of the data into a validation set.



TODO
I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.