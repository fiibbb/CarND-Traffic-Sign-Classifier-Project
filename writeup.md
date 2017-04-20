#**Traffic Sign Recognition** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/label_dist.png "Visualization"
[image2]: ./test_images/test1.jpg "Traffic Sign 1"
[image3]: ./test_images/test2.jpg "Traffic Sign 2"
[image4]: ./test_images/test3.jpg "Traffic Sign 3"
[image5]: ./test_images/test4.jpg "Traffic Sign 4"
[image6]: ./test_images/test5.jpg "Traffic Sign 5"

###Data Set Summary & Exploration

####Basic summary

The code for this step is contained in the first and second code cell of the IPython notebook.  

I calculated the summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of test set is 12630
* The shape of a traffic sign image is (32,32,3)
* The number of unique classes/labels in the data set is 43

####Visualization

The code for this step is contained in the second code cell of the IPython notebook.  

Here is an exploratory visualization of the data set. It is a bar chart showing how the data:  

![alt text][image1]

###Design and Test a Model Architecture

####Preprocessing

The code for this step is contained in the fourth code cell of the IPython notebook.

For preprocessing, I chose to first grayscale the images. Traffic signs are designed to be clearly identifiable even without full RGB color. Thus, turning images into grayscale will help the model ignore the irrelevant color information. Then I normalize the grayscale intensity from `0 - 255` to `-1.0 - 1.0`. This is a common normalization technique used in preprocessing image data that will further help the model find the generalized patterns.


####Model Architecture

The code for my final model is located in the fifth cell of the ipython notebook. 

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 grayscale image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling     	| 2x2 stride, valid padding, outputs 14x14x6	|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 10x10x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
| Flatten	    | 			outputs 1x400				|
| Fully connected		| outputs 1x200        									|
| Droputs		|         									|
| Fully connected		| outputs 1x100        									|
| Droputs		|         									|
| Fully connected		| outputs 1x43        									|
 
####Training, Validation and Testing

The code for training, validation and testing is contained in the 6th code cells.

I chose to base my model on the standard LeNet model. However, I tweaked the last two fully connected layers to be slightly larger than LeNet's. This was added because I realized my accuracy seem to hit a plateau around 85%. Augmenting the fully connected layers seem to alleviate the issue and gave an accuracy of >90%. Another issue was overfitting -- after augmenting the layers, the model showed test accuracy significantly lower than validation accuracy (<90% test accuracy while >95% validation accuracy). Thus, I added dropout layers after these fully connected layers to force the model generalize its predictions.

I chose to use the Adam optimizer. The main advantage of Adam optimizer here is that it will automatically adjust the learn rate, so I only need to tune the inital learn rate. I ended up with 10 epochs, because the accuracy is high enough after roughly 5-7 epochs, and the marginal gain after that seem minimal. The dataset is also randomly shuffled in each epoch, and fed to the model in batch of 128 to reduce memory pressure.

####Final result

The code for calculating the accuracy of the model is located in the 6th cell of the Ipython notebook.

My final model results were:  

* validation set accuracy of 0.958  
* test set accuracy of 0.939 

###Test the Model on New Images

####New images

Here are five German traffic signs that I found on the web:

![alt text][image2] ![alt text][image3] ![alt text][image4] 
![alt text][image5] ![alt text][image6]

Note that the images shown here are full scale images, where as since the model only expectes 32x32 images, they are resized as shown in the IPython notebook. The resolution decrease may pose difficulty to the model when it comes to prediction, especially for 1st and 4th image, where the shape of the person is largely distorted, although still recognizable to human eyes.

####Prediction

The code for making predictions on my final model is located in the 6th cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Traffic Sign 1      		| 27 (Pedestrians)   									| 
| Traffic Sign 2     			| 3 (Speed limit 60km/h) 										|
| Traffic Sign 3					| 38 (Keep right)											|
| Traffic Sign 4	      		| 28 (Children crossing)					 				|
| Traffic Sign 5			| 18 (General caution)      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This is significantly lower than the test accuracy of 93.3%. It is worth noting that the sample is small here (5 images), which may contribute to the lower than expected accuracy.

The model showed very high confidence in four image predictions (sign 1, 2, 4, 5), all above 99%, while the prediction for sign 3 is 83.7%. This aligns well with the fact that it made a wrong prediction on the 3rd image.
