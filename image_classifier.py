from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from PIL import Image
import numpy as np
import sys
import os

def load_image(image_path):
    #Load and preprocess the image.
    try:
        image = Image.open(image_path)
        image = image.resize((224, 224))
        image = np.array(image)
        image = preprocess_input(image)
        image = np.expand_dims(image, axis=0)
        return image
    except FileNotFoundError:
        print(f"Error: File not found at '{image_path}'. Please provide a valid file path.")
        sys.exit(1)
    except IOError:
        print(f"Error: Unable to open file at '{image_path}'. Please provide valid image file.")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading image: {e}")
        sys.exit(1)

def classify_image(image_path, model):
    #Classify the image and print the result.
    image = load_image(image_path)
    predictions = model.predict(image)
    decoded_predictions = decode_predictions(predictions, top=1)[0]
    return decoded_predictions

def main():
    if len(sys.argv) != 2:
        print("Usage: python image_classifier.py <path_to_image>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    if not os.path.isfile(image_path):
        print(f"File not found: {image_path}")
        sys.exit(1)
    
    # Load the pre-trained MobileNetV2 model
    model = MobileNetV2(weights='imagenet')
    
    # Classify the image
    predictions = classify_image(image_path, model)
    
    # Output the result
    for pred in predictions:
        print(f"Predicted: {pred[1]} (Confidence: {pred[2] * 100:.2f}%)")

if __name__ == "__main__":
    main()
