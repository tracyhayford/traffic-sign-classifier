# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

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

[image1a]: ./writeup_graphics/output_8_1.png "Training Data Set"
[image1b]: ./writeup_graphics/output_8_3.png "Test Data Set"
[image1c]: ./writeup_graphics/output_8_5.png "Validation Data Set"
[image2]: ./writeup_graphics/output_12_1.png "Normalization"
[image4]: ./TrafficSignsForTesting/70mph.jpg "Traffic Sign 1"
[image5]: ./TrafficSignsForTesting/Stop_Sign.jpg "Traffic Sign 2"
[image6]: ./TrafficSignsForTesting/no-passing-no-passing-stock-images_csp3940925.jpg "Traffic Sign 3"
[image7]: ./TrafficSignsForTesting/RoadWork.jpg "Traffic Sign 4"
[image8]: ./TrafficSignsForTesting/turn-right-sign-regulatory-signs-turn-right-ahead-traffic-sign-stock-photo_csp38482526.jpg "Traffic Sign 5"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

This is the project report or write up.  My code is in the project jupyter notebook "Traffic_Sign_Classifier.ipynb" and here is a link [project code](./Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library (np.shape() function) to calculate summary statistics of the traffic signs data set.  Here is my output:

	Number of training examples = 34799
	Number of testing examples = 12630
	Number of validation examples = 4410
	Image data shape = [32, 32]
	Number of classes = 43
	
	Sample signs [0] & [5]:
	['0', 'Speed limit (20km/h)']
	['5', 'Speed limit (80km/h)']

#### 2. Include an exploratory visualization of the dataset.

I chose to provide exploratory visualization by displaying 20 randomly selected images from each data set. 

Training
![alt text][image1a]

Test
![alt text][image1b]

Validation
![alt text][image1c]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As suggested, standard preprocessing techniques are converting the images to grayscale and normalizing the image data.  Normalization produces a single number scaled from -1 to 1 for each pixel in the image which represents the grayscale value.


Here is an example of an original image and an augmented image:

![alt text][image2]

The difference between the original data set and the augmented data set is that the augmented data set color depth is 1 (grayscale) instead of 3 (RGB) which should reduce processing requirements.  Additionally, the normalization process appeared to improve brightness and contrast as seen in the first and second images.

To provide easy to understand code, I defined the function "normalize_group" which takes as input two booleans - "gray" and "normalz" which select the normalization options.  Grayscaling is accomplished by simply dividing each R, G and B pixel value by 3 and summing these together to create one element out of the three.  Value normalization is performed by subtracting 128 and dividing by 128 (since the values range from 0 to 255). 


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

As suggested, I started with the LeNet lab LeNet model.  Several layers were added as the model didn't produced the required accuracy and fitting.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 grayscale image 						| 
| Convolution 5x5     	| 1x1 stride, same padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6	 				|
| Convolution 5x5     	| 1x1 stride, same padding, outputs 10x10x16 	|
| RELU					|												|
| Dropout		      	| 								 				|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16	 				|
| Flatten				| outputs 400 (1D)								|
| Fully connected		| outputs 120  									|
| RELU					|												|
| Dropout		      	| 								 				|
| Fully connected		| outputs 84  									|
| RELU					|												|
| Fully connected		| outputs n_classes (43)						|
|						|												|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I set up a "training_operation" in TensorFlow.  This consists of using the LeNet model to generate logits.  The cross entropy is calculated using the TF function "nn.cross_entropy_with_logits" and loss calculated using the TF function "reduce_mean".  The Adam optimizer is used with a specified learning rate and the loss is minimized using the optimizer's minimze function.

Additionally, "accuracy operation" was coded to compare predictions with one_hot truth values from the data set.

An "evaluate" function is defined that processes images through a batching process and calculates an average accuracy over the batches.  This is used to calculate the accuracy of a trained model in predicting test and validation sign types.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.



My final model results were:

* validation set accuracy of 0.947 (peaked at 0.955 in Epoch 13) 
* test set accuracy of 0.931

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The third image might be difficult to classify because the sign isn't centered in the image field and it doesn't fill the image frame.  This results in a less distinct image to process when resized to 32x32.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 70 km/h				| Speed Limit (70 km/h) 						|
| Stop Sign      		| Stop sign   									| 
| No passing			| Roundabout mandatory							|
| Road work	      		| Bumpy road					 				|
| Right turn			| Keep left		    							|


The model was able to correctly guess only 2 of the 5 traffic signs, which gives an accuracy of 40%. The accuracy of the test set was much higher possibly due to the image quality or selection or some other code error.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

Here's the output of the code:

	INFO:tensorflow:Restoring parameters from ./lenet
	Top 5 Softmax probabilities and sign types
	  Test Image  1  (prob):  [ 0.06078682  0.02236222  0.02236222  0.02236222  0.02236222]
	  Sign Types: [ Speed limit (70km/h) Speed limit (30km/h) Speed limit (20km/h) Speed limit (50km/h) Speed limit (60km/h) ]
	  Test Image  2  (prob):  [ 0.05922854  0.02266321  0.02249699  0.02246671  0.02241565]
	  Sign Types: [ Stop Road work Bumpy road Keep right Bicycles crossing ]
	  Test Image  3  (prob):  [ 0.04759689  0.02634168  0.02409337  0.02285414  0.02265665]
	  Sign Types: [ Roundabout mandatory Speed limit (70km/h) Traffic signals Priority road General caution ]
	  Test Image  4  (prob):  [ 0.0579031   0.02347191  0.02245613  0.02241659  0.02241286]
	  Sign Types: [ Bumpy road Road work Wild animals crossing General caution Traffic signals ]
	  Test Image  5  (prob):  [ 0.04549969  0.03016457  0.02260444  0.02256666  0.02255167]
	  Sign Types: [ Keep left Turn right ahead No passing Stop Speed limit (70km/h) ]

The values shown for the top probabilities are unexpectedly low.  A review of the complete output of the softmax shows that all 43 elements have values no lower than approximately 0.02.  I would have expected almost all of the probability values to be very low except 5 or 6 with higher probabilities including only 1 or 2 with particularly high probabilities.  I wasn't able to identify a problem with the model or the prediction algorithm that I thought would produce this result.

For the first image, the model is three times more sure that this is a Speed limit (70km/h) sign(probability of 0.061) than any other sign, and the image does contain that sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .061         			| Speed limit (70km/h)  						| 
| .022     				| Speed limit (30km/h)							|
| .022					| Speed limit (20km/h)							|
| .022	      			| Speed limit (50km/h)			 				|
| .022				    | Speed limit (60km/h) 							|


For the second image, the model is three times more sure that this is a stop sign (probability of 0.059) than any other sign, and the image does contain that sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .059         			| Stop    										| 
| .023     				| Road work										|
| .022					| Bumpy road									|
| .022	      			| Keep right					 				|
| .022				    | Bicycles crossing    							|

For the third image, the model is about twice as sure that this is a Roundabout mandatory sign (probability of 0.047) than any other sign, but the image is really a No Passing sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .047         			| Roundabout mandatory   						| 
| .026    				| Speed limit (70km/h)							|
| .024					| Traffic signals								|
| .023	      			| Priority road					 				|
| .023				    | General caution      							|

For the fourth image, the model is nearly three times more sure that this is a Bumpy road sign (probability of 0.057), but the image contains a Road work sign.  However, that is the model's 2nd choice. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .057         			| Bumpy road       								| 
| .023     				| Road work										|
| .022					| Wild animals crossing							|
| .022	      			| General caution				 				|
| .022				    | Traffic signals      							|

For the fifth image, the model is sure that this is either a Keep left or Turn right ahead sign (probability of 0.045 + 0.030) and while the top pick isn't correct (Keep left), the image does contain a Turn right ahead sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .045         			| Keep left       								| 
| .030     				| Turn right ahead								|
| .023					| No passing									|
| .023	      			| Stop							 				|
| .023				    | Speed limit (70km/h) 							|


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

Not attempted
