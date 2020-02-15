# Traffic-Light-Detection-and-Classification

This project is one part of the Udacity Self-Driving Car Capstone Project. Within a simulator, the car will drive around a track and needs to stop at intersections with red traffic lights. Therefore a process is needed to identify traffic lights in image data and classify the light color. The intersections in the simulator are more simplistic than a real world example, since there are only 3 states (Red, Yellow, Green) and no arrows for example. Therefore only one identified color refers to all lanes in the simulator. Nevertheless one focus is also the speed of the model: the whole process should run multiple times per s. 

![png](media/classification_pipeline.png)

Project goals:
- use an existing tensorflow object detection model to accuratly identify traffic lights in images
- generate training data with object detecion and [BOSCH Dataset](https://hci.iwr.uni-heidelberg.de/node/6132)
- implement and train CNN on created data
- implement full pipeline from image input to classification

## 1. Object detection for traffic lights

The first objective in this project is to successfully extract traffic light objects from image input to a standardized format. For this task it's best to refer to already trained models. Well suited for this task are the object detection models in the [Tensorflow detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md). 

These models are pre-trained on the [COCO
dataset](http://mscoco.org), the [Kitti dataset](http://www.cvlibs.net/datasets/kitti/),
the [Open Images dataset](https://github.com/openimages/dataset), the
[AVA v2.1 dataset](https://research.google.com/ava/) and the
[iNaturalist Species Detection Dataset](https://github.com/visipedia/inat_comp/blob/master/2017/README.md#bounding-boxes) and already feature a label for traffic lights (10). Therefore any of these models could be used out of the box. 

Since the target project is within an embedded system, speed will matter. Considering this, I used the ssd_mobilenet_v1_0.75_depth_coco, which on reference has the fastest performance. 

First I initilize the tl_detection with the already existing frozen inference graph and link the input and output tensors of the graph with class variables. The input tensor in this graph is an image of any size. The outputs are then boxes around the object, score (certainty) and the corresponding class. The outputs are ordered by score, which means high certainty is also high up on the list. I implemented a filter on the score to be at least above 0.5 certainty.

![png](media/object_detection_pipeline.png)

A small showcase on how the object detection works. The Object detection takes the image as input and stores an image with boxes around traffic lights as well as the image with highest certainty cropped. This format (32,14) will also be the format for the CNN later build to classify the light color. Within the udacity simulator all traffic lights have the same color, therefore we could actually stop and break the loop in the embedded system after we have detected one (see image 2 from simulator). The boxes around the image and the original will not be needed in the final implementation.

## 2. Generating Training data

Now after we have a piece of code to return a cropped and standardized box of a traffic light, we can start to implement a classification model. First thing that needs to be done is to look for some training data. I came across the [BOSCH Dataset](https://hci.iwr.uni-heidelberg.de/node/6132), which features around 5000 consecutive images of driving and already labeled and detected traffic light boxes. So to create our training data, we just need to download it and iterate over all the images to see if they have a traffic light with label in them. I also check that the size of the traffic light is big enough (<25 pixel) to ensure it is actually visible. I end up with ~3600 images of traffic lights with my target size.

    3625

![png](media/training_data.png)


There are a cupple more things that need to be done before continuing with the model. First is the label: the BOSCH dataset features more labels that we actually need, e.g. the arrows with color which are also distinguished. Since the udacity simulator does not have these features, we can safely drop these in the label as well. Additionally we will have to check the balance of the labels, so we can adjust them if needed. 

    {'G': 2041, 'R': 1301, 'Y': 89, 'o': 194}

As can be seen, the dataset is heavily skewed towards Green and Red, which is of course rational when thinking about how traffic lights work - yellow is just a transition phase for a short period of time. The Dataset also features another class 'o', which is basically unknown/turned off. Since this state is also present in udacity simulator, we will keep it. 

Much later in the process I noticed that my model was able to deal much better with the classification when increasing the counts for Yellow & unknown label. Therefore I implemented a random choice skript to generate my training data with weights different than the original distribution. The data used for training features following distributions: {'G': 1149, 'R': 1241, 'Y': 304, 'o': 306}.

    {'G': 1197, 'R': 1211, 'Y': 289, 'o': 303}

Lastly, I change the string class for an integer for the CNN and save everything for the model. My original data can be found in the repo.

## 3. Network Architecture

I choose a similar CNN architecture as NVIDIA on their [End to End Learning for Self-Driving Cars](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) Network and made some adjustments. For once I decreased the size due to a much simpler model. Since we are using (not really much) image data, it is favorable to use tensorflows image data generator to add some random augmentation to our images. This will hopefully improve the models ability to generalize better and deal with some distorted observations that might come from the cars image data.

- Input Layer (32,14,3)
- Normalization Layer (centered around 0)
- Conv2D (16, Kernel = (3,3), Strides = (2,2), Activation = Rectified Linear)
- MaxPooling2D
- Dropoout (0.3)
- Conv2D (32, Kernel = (3,3), Strides = (2,2), Activation = Rectified Linear)
- MaxPooling2D
- Dropoout (0.3)
- Conv2D (64, Kernel = (3,3), Strides = (2,2), Activation = Rectified Linear)
- Dropoout (0.3)
- Dense (100, Activation = Rectified Linear)
- Dropoout (0.3)
- Dense (20, Activation = Rectified Linear)
- Dropoout (0.2)
- Dense (4, Activation = Softmax)


## 4. Model Results

The model was trained with a learning rate of 1e-3 and a decay of 1e-5. Since the classification model is quite simple, I noticed that this combination got quite good results. The model is trained for 20 epochs, after which it stagnates and doesnt improve any further.

    Epoch 0
    Train on 2400 samples
    2400/2400 [==============================] - 1s 232us/sample - loss: 1.2140 - acc: 0.4150
    Epoch 1
    Train on 2400 samples
    2400/2400 [==============================] - 0s 136us/sample - loss: 0.7950 - acc: 0.7121
    Epoch 2
    Train on 2400 samples
    2400/2400 [==============================] - 0s 142us/sample - loss: 0.4783 - acc: 0.8029
    Epoch 3
    Train on 2400 samples
    2400/2400 [==============================] - 0s 146us/sample - loss: 0.3259 - acc: 0.8742
    Epoch 4
    Train on 2400 samples
    2400/2400 [==============================] - 0s 170us/sample - loss: 0.2809 - acc: 0.9121
    Epoch 5
    Train on 2400 samples
    2400/2400 [==============================] - 0s 140us/sample - loss: 0.2250 - acc: 0.9308
    Epoch 6
    Train on 2400 samples
    2400/2400 [==============================] - 0s 148us/sample - loss: 0.2140 - acc: 0.9312
    Epoch 7
    Train on 2400 samples
    2400/2400 [==============================] - 0s 131us/sample - loss: 0.2259 - acc: 0.9304
    Epoch 8
    Train on 2400 samples
    2400/2400 [==============================] - 0s 144us/sample - loss: 0.2195 - acc: 0.9375
    Epoch 9
    Train on 2400 samples
    2400/2400 [==============================] - 0s 163us/sample - loss: 0.1963 - acc: 0.9438
    Epoch 10
    Train on 2400 samples
    2400/2400 [==============================] - 0s 138us/sample - loss: 0.1911 - acc: 0.9433
    Epoch 11
    Train on 2400 samples
    2400/2400 [==============================] - 0s 139us/sample - loss: 0.2056 - acc: 0.9442
    Epoch 12
    Train on 2400 samples
    2400/2400 [==============================] - 0s 129us/sample - loss: 0.1929 - acc: 0.9508
    Epoch 13
    Train on 2400 samples
    2400/2400 [==============================] - 0s 143us/sample - loss: 0.1891 - acc: 0.9463
    Epoch 14
    Train on 2400 samples
    2400/2400 [==============================] - 0s 138us/sample - loss: 0.1932 - acc: 0.9504
    Epoch 15
    Train on 2400 samples
    2400/2400 [==============================] - 0s 136us/sample - loss: 0.1794 - acc: 0.9483
    Epoch 16
    Train on 2400 samples
    2400/2400 [==============================] - 0s 144us/sample - loss: 0.1773 - acc: 0.9479
    Epoch 17
    Train on 2400 samples
    2400/2400 [==============================] - 0s 138us/sample - loss: 0.1691 - acc: 0.9525
    Epoch 18
    Train on 2400 samples
    2400/2400 [==============================] - 0s 137us/sample - loss: 0.1736 - acc: 0.9479
    Epoch 19
    Train on 2400 samples
    2400/2400 [==============================] - 0s 133us/sample - loss: 0.1704 - acc: 0.9542
    

The classification report with test data show a general accuracy of 0.96. The most trouble for our model is actually the classification of unkown. Since the original distribution of this class is so low, the accuracy in a real world example would probably be better. In general I believe this is a solid model.

                  precision    recall  f1-score   support
    
               G       0.99      1.00      0.99       233
               R       0.97      0.94      0.95       243
               Y       0.98      0.84      0.91        63
               o       0.76      0.93      0.84        61
    
        accuracy                           0.95       600
       macro avg       0.92      0.93      0.92       600
    weighted avg       0.96      0.95      0.95       600
    
    

## 5. Detection & Classification Pipeline

Now I have to bring all individual pieces together and want to implement them within one class. This class will be feed raw image data, it will store augmented images and return the final prediction of the label. 

For building this we need:
- both of the saved models to be initialized at start
- pipeline for image feed (detection & classification)
- map prediction back to label

    Classification: G after 2.885s  
    
    Classification: Y after 0.058s  
    
    Classification: Y after 0.06s   
    
    Classification: G after 0.072s  
    
    Classification: R after 0.06s
    
![png](media/classification_pipeline.png)


As can be seen the first prediction takes some time to complete. The image from the simulator (2nd) is then afterwards completed after 0.06s. Therefore it could be run multiple times per second within the simulator to classify the state. One might also think of adding some additional logic to the classifications. 

For example the order of traffic light colors is always G -> Y -> R -> Y -> G. So when we successfully identified the inital image we can compare this information with the following observations to exclude some unlogical classifications or will already know something about the future if a light turns yellow. For example we could calculate the distance to the traffic light, to see if we will still make it in time or if we should treat that as a red light and stop. Similary, when the traffic light turns yellow from red, we know its already safe to accelerate. 
