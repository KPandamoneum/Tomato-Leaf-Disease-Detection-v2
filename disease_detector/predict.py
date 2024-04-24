import numpy as np
import cv2
from tensorflow.keras.models import load_model


class_dict = {
    'Tomato___Bacterial_spot': 0,
    'Tomato___Early_blight': 1,
    'Tomato___Late_blight': 2,
    'Tomato___Leaf_Mold': 3,
    'Tomato___Septoria_leaf_spot': 4,
    'Tomato___Spider_mites Two-spotted_spider_mite': 5,
    'Tomato___Target_Spot': 6,
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': 7,
    'Tomato___Tomato_mosaic_virus': 8,
    'Tomato___healthy': 9
}


model = load_model("vgg19_l.keras")

def prepare_image(filepath):
    """Process the image ucing OpenCV and retun an array
    """
    img_array = cv2.imread(filepath, cv2.IMREAD_COLOR)
    img_array = img_array / 255
    new_array = cv2.resize(img_array, (128, 128))
    return new_array.reshape(-1, 128, 128, 3)


def prediction_cls(prediction):
    """Process the numeral output to give the corresponding class in string
    """
    for key, clss in class_dict.items():
        if np.argmax(prediction) == clss:
            return key



def prediction_main(filepath, model=model):
    img_data = [prepare_image(filepath)]
    prediction = model.predict(img_data)
    
    return np.max(prediction) * 100, prediction_cls(prediction)
