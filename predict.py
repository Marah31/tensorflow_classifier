import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import argparse
import json
from PIL import Image

# feature extractor defined globally
URL = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"
feature_extractor = hub.KerasLayer(URL, input_shape=(224, 224, 3), trainable=False)

@tf.keras.utils.register_keras_serializable()
def feature_extraction(x):
    return feature_extractor(x)

def process_image(image):
    image = tf.image.resize(image, (224, 224))
    image = tf.cast(image, tf.float32) / 255.0
    return image

# predict function
def predict(image_path, model, top_k=5):
    image = Image.open(image_path)
    image = np.asarray(image)
    image = process_image(image)
    image = np.expand_dims(image, axis=0)
    predictions = model.predict(image)
    top_probs = np.sort(predictions[0])[-top_k:][::-1]
    top_classes = np.argsort(predictions[0])[-top_k:][::-1]
    return top_probs, top_classes

def main():
    parser = argparse.ArgumentParser(description='Predict flower image class')
    parser.add_argument('image_path', type=str, help='Path to the image file')
    parser.add_argument('model', type=str, help='Path to the saved model')
    parser.add_argument('--top_k', type=int, default=5, help='Return the top K most likely classes')
    parser.add_argument('--category_names', type=str, default=None, help='Path to a JSON file mapping labels to flower names')
    
    args = parser.parse_args()

    # load the model
    model = tf.keras.models.load_model(args.model, custom_objects={'feature_extraction': feature_extraction, 'KerasLayer': hub.KerasLayer})

    probs, classes = predict(args.image_path, model, args.top_k)

    
    if args.category_names:
        with open(args.category_names, 'r') as f:
            class_names = json.load(f)
        classes = [class_names.get(str(cls), "Unknown") for cls in classes]
    # print the results
    print(f"Top {args.top_k} Predictions:")
    for prob, cls in zip(probs, classes):
        print(f"{cls}: {prob*100:.2f}%")

if __name__ == "__main__":
    main()
