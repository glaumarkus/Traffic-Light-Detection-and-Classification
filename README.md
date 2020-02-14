# Traffic-Light-Detection-and-Classification
short project to detect traffic lights from image input and classify color

This project is one part of the Udacity Self-Driving Car Capstone Project. Within a simulator, the car will drive around a track and needs to stop at intersections with red traffic lights. Therefore a process is needed to identify traffic lights in image data and classify the light color. The intersections in the simulator are more simplistic than a real world example, since there are only 3 states (Red, Yellow, Green) and no arrows for example. Therefore only one identified color refers to all lanes in the simulator. Nevertheless one focus is also the speed of the model: the whole process should run multiple times per s. 

Project goals:
- use an existing tensorflow object detection model to accuratly identify traffic lights in images
- generate training data with object detecion and BOSCH dataset (can be found here)
- implement and train CNN on created data
- implement full pipeline from image input to classification

Dependencies:
tensorflow == 1.15

1. Object detection for traffic lights

2. Generating Training data

3. Network Architecture

4. Model Results

5. Detection & Classification Pipeline
