# imageclassifier

The main objective of this project is to Create a Python application that uses a pre-trained deep learning model to classify images into various categories.



## Tech Stack used

Python, tensorflow, numpy, CLI, MobileNetV2, PIL



## Environment Variables

To run this project, you will need to add the following environment variables to your .env file

## Installation

You can install the required libraries using pip:

```bash
  pip install tensorflow
  pip install numpy
```
After Installation of Libraries we need to import required packages by using follwing:
```bash
import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from PIL import Image
import sys
import os

```

## Discription

* #### Tensorflow: TensorFlow is a popular framework of machine learning and deep learning. It is a free and open-source library which developed by Google Brain Team. It is entirely based on Python programming language and use for numerical computation and data flow, which makes machine learning faster and easier.
* #### numpy: numpy is also open and source framework which is used for done numerical caluculations on arrays.
* #### PIL:It is an open source library specifically designed for image processing via Python.
* #### keras:Keras being a model-level library helps in developing deep learning models by offering high-level building blocks.
* #### MobileNetV2:MobileNet-v2 is a convolutional neural network. You can load a pretrained version of the network trained on more than a million images from the ImageNet database






## Implementation

### Functions
1.load_image(image_path):

* Loads and preprocesses the image  from the given file path.
* Resizes the image to 224x224 pixels.
* Converts the image to a format that the model can understand.

2.classify_image(image_path, model):

* Uses load_image to preprocess the image.
* Uses the MobileNetV2 model to predict what the image is.
* Decodes the predictions to human-readable labels and returns the top prediction.

3.main():

* Checks if the correct number of command-line arguments are provided.
* Verifies that the provided image file path is valid.
* Loads the MobileNetV2 model with pre-trained weights.
* Classifies the image and prints the top prediction along with its confidence score.


## Execution

### How to Use the Script
* Save the script to a file, e.g., image_classifier.py.

* Open a terminal in cmd and go to the directory containing the script.

* Run the script with the path to the image you want to classify: 

```bash
  python image_classifier.py <Image_path>
```
### Sample Execution

```bash
  python image_classifier.py sampleimage.jpg
```

### Sample Output

After sucessfull running of this code we can see the Output as see like this.

```bash
  Predicted: bee_eater (Confidence: 85.15%)
```
