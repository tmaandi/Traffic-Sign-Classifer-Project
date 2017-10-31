# Traffic Sign Classifier
# Brief Overview


This project uses the LeNet Deep Learning Model architecture to train a convolutional neural network based classifier to classify/recognize 43 different German Traffic Signs. Various data-preprocessing techniques (like normalization, mean zeroing etc), followed by a LeNet neural network augumented by regularization techniques (like Dropout, learning rate decay) are used in this project. In addition to the pre-segregated test data, the trained network is also tested on 5 randomly chosen images from the Web.



[//]: # (Image References)

[image1]: ./DataVisualization.png "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"


[image1]: ./DataVisualization.png "Visualization"

### Data Set Summary & Exploration

I primarily used numpy and matplotlib libraries to calculate summary statistics and visualize the traffic
signs data set:

* The size of training set is 34799 samples
* The size of the validation set is 4410 samples
* The size of test set is 12630 samples
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

Here is an exploratory visualization of the data set. The first image is a sample image randomly picked from the dataset (The class index can be referred to the .csv database to identify the image label). A pie chart on the right shows the distribution of data among training, validation and test data sets. Below is a bar chart showing how the data is distributed among 43 different classes.

![alt text][image1]

### Design and Test a Model Architecture
[image2]: ./Normalization.png "Effect of Normalization and mean-zeroing"

As a first step, I decided to nomalize the image data and zeroed the mean as it reduces the spread of the feature data and thus makes the learning process less prone to divergence. (I also tried Grayscaling but found that network trained and performed better without grayscaling)

Below is the image which shows the effect of normaliztion and mean-zeroing on an image
![alt text][image2]

I also tried data augumentation by appending the training data set with images rotated by 90 degree but it did not show any significant improvement but did come with an increase in the training time so that was not implemented.

#### Model Architecture

My final model consisted of the following layers: A standard LeNet5 Architecture

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6. 	|
| RELU					|			                                    |                                               
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				    |
| Dropout               | Keep Probability: 75 %                        |
| Convolution 5x5	    | 1 x 1 stride, valid padding, outputs 10x10x16 |      			
| RELU					|			                                    |                                               
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				    |
| Dropout               | Keep Probability: 75 %                        |
| Fully connected		| Input = 400. Output = 120.        			|	
| RELU			     	|            									|
| Fully connected		|Input = 120. Output = 84						|
| RELU					|												|
| Fully connected       |Input = 84. Output = 43                        |


[image3]: ./test_images_web/road_work.jpg "Road Work"
[image4]: ./test_images_web/slippery_road.jpg "Slippery Road"
[image5]: ./test_images_web/speed_limit_120.jpg "Speed Limit"
[image6]: ./test_images_web/stop.png "Stop Sign"
[image7]: ./test_images_web/yield.png "Yield"
[image8]: ./test_images_web/right_of_way_at_next_intersection.png "Right of Way"

#### 3. Model Training

To train the model, I used AdamOptimizer. The initial validation accuaracy from LeNet was about 87%. The batch size was gradually decreased from 128 to 16 and the number of epochs were increased to 15. Though it increased the training time but significantly helped with the validation accuracy.

#### 4. Iterative Improvisations/Additions
The first modification I did was adding the Dropout layer which helped the model to have a more generalized training and not overfit for some particular set of nodes. Also, I observed that model's validation accuaracy was oscllating about a value after a few epochs. It appeared that a high learning rate was causing this. So I introduced the learning rate decay as a function of validation accuracy but it didn't solve the problem that well. Then I tried learning rate decay based on the epochs, i.e, gradually decreasing the learning rate with increasing number of epochs. This helped. I did not play with the hidden layer depths which might or might not have resulted in better performance.


My final model results were:

* validation set accuracy of 95.7 % 
* test set accuracy of 94.4 %

### Testing the Model on New Images

Here are five German traffic signs that I chose from the web:

![alt text][image3] ![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7]

The first image might be difficult to classify because of the following reasons: A majority of the other 'Road Work' images do not have this text written over them but only the road sign; secondly, the angle of the plane of this traffic sign is a bit tilted in this image, which has a skewed 2-D projection which network might be finding difficult to identify correctly;thirdly, the wrinkles in the signboard itself might transform themseleves into a 'feature' when fed to the network and may lead to miss-classification.
In addition to this, the watermark in the 'Roadwork' and 'Slippery Road' images and small size and low contrast of 'Stop' sign image are also a cause of concern.


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of 94.4%.


Here are the results of the prediction:

| Image			                |     Prediction	        					| 
|:-----------------------------:|:---------------------------------------------:| 
| Road Work              (25)	| Right of way at the next intersection (11)    | 
| Slippery Road	         (23)	| Slippery Road									|
| Speed Limit (120 km/h) (8)	| Speed Limit (120 km/h)						|
| Stop Sign	             (14)	| Stop Sign 					 				|
| Yield	     	         (13)	| Yield	            							|


The network confused the 'Roadwork' sign with 'Right of the way at the next intersection' as shown below. Maybe becuase of the tilted plane of the image, the network confused the centre sign with the centre sign like the one in this image:

![alt text][image8]



The code for making predictions on my final model is located in the 10th cell of the Ipython notebook.

For the first image, the model is completely unable to classify that this is a "Road Work" sign (probability not in top 5) as mentioned above. The top five soft max probabilities for this sign were:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.784        			| 11: Right of way at the next intersection  	| 
| 0.152    				| 21: Double Curve 								|
| 0.030					| 28: Children Crossing		     				|
| 0.009	      			|  2: Speed Limit (50 km/h)		 				|
| 0.008				    |  5: Speed Limit (80 km/h)    					|


The Second image was classified correctly. The top 5 softmax probabilities for this image are:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.000        			| 23: Slippery Road  	                        | 
| 0.000    				| 19: No Entry				     				|
| 0.000					| 10: Speed Limit (120 km/h)     				|
| 0.000	      			|  2: Speed Limit ( 20 km/h)	 				|
| 0.000				    | 29: Pedestrians           					|

The Third image was classified correctly. The top 5 softmax probabilities for this image are:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.903        			|  8: Speed limit (120km/h)                 	| 
| 0.005    				|  0: Speed limit (20km/h) 						|
| 0.004					|  7: Speed limit (100km/h)		     			|
| 0.000	      			|  2: Speed limit (50km/h)		 				|
| 0.000				    |  4: Speed limit (70km/h)    					|

The Fourth image was classified correctly. The top 5 softmax probabilities for this image are:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.943        			| 14: Stop  	                                | 
| 0.029    				| 12: Priority road 							|
| 0.000					|  1: Speed limit (30km/h)		     			|
| 0.000	      			| 13: Yield		 				                |
| 0.000				    |  3: Speed limit (60km/h))    					|

The Fifth image was classified correctly. The top 5 softmax probabilities for this image are:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.000        			| 13: Yield                                 	| 
| 0.000    				| 12: Priority road 							|
| 0.000					|  1: Speed limit (30km/h)	     				|
| 0.000	      			|  9: No passing		 			        	|
| 0.000				    | 15: No vehicles            					|