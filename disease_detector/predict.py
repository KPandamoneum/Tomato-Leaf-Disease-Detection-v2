import numpy as np
import cv2
from tensorflow.keras.models import load_model #type: ignore


class_dict = {
    'Bacterial Spot': 0,
    'Early Blight': 1,
    'Late Blight': 2,
    'Leaf Mold': 3,
    'Septoria Leaf Spot': 4,
    'Spider Mites Two-Spotted Spider mite': 5,
    'Target Spot': 6,
    'Tomato Yellow Leaf Curl Virus': 7,
    'Tomato Mosaic Virus': 8,
    'Healthy': 9
}


descriptons = [
    "Caused by bacteria, this disease creates small, dark spots on leaves and fruit. It thrives in cool, wet weather. Unfortunately, there's no cure, but planting during warm weather and avoiding overhead watering can help prevent its spread.",
    "The most common tomato leaf spot disease. It's caused by fungus and shows up as brown or gray spots on leaves and stems in hot, humid areas. Choose resistant tomato varieties, remove infected leaves, and use fungicide when necessary.",
    "A serious fungal disease causing rapid wilting, brown lesions on leaves and stems, and watery rot on fruit. It thrives in cool, wet weather. Rotate crops, remove infected plant debris, and use fungicide preventatively, especially in cool, damp weather.",
    "Fungal disease that thrives in cool, humid weather. It causes fuzzy gray patches on the undersides of leaves and can lead to fruit rot. Improve air circulation, avoid overhead watering, use resistant varieties, and apply fungicide as needed.",
    "Fungal disease causing brown or gray circular spots on leaves with dark centers. It can lead to defoliation in severe cases. Rotate crops, remove infected plant debris, water at the base of the plant, and use fungicide if necessary.",
    "Tiny, sucking pests that cause yellow stippling on leaves and webbing on the undersides. They thrive in hot, dry weather. Strong blasts of water to remove mites, insecticidal soap, or neem oil spray.",
    "Fungal disease causing brown or black spots with concentric rings on leaves and stems. It can also affect fruit. Rotate crops, remove infected plant debris, water at the base of the plant, and use fungicide when necessary.",
    "Virus transmitted by whiteflies that stunts plant growth, yellows leaves, and curls leaf edges. Control whiteflies to prevent the spread of the virus. No cure exists for infected plants.",
    "Virus causing stunted growth, mottled leaves, and distorted fruit. It can be spread by mechanical means or through infected tools or transplants. No cure exists, but you can prevent spread by disinfecting tools and removing infected plants.",
    "This refers to a tomato plant free of any diseases or pests.",
]


# model = load_model("model_best.h5")
model = load_model("vgg19.h5")

def prepare_image(filepath):
    """Process the image ucing OpenCV and retun an array
    """
    img_array = cv2.imread(filepath, cv2.IMREAD_COLOR)
    img_array = img_array / 255
    # new_array = cv2.resize(img_array, (224, 224))
    # return new_array.reshape(-1, 224, 224, 3)
    new_array = cv2.resize(img_array, (128, 128))
    return new_array.reshape(-1, 128, 128, 3)


def prediction_cls(prediction):
    """Process the numeral output to give the corresponding class and description in strings
    """
    prediction_ = np.argmax(prediction)
    predicted_class = None
    for key, clss in class_dict.items():
        if prediction_ == clss:
            predicted_class = key
    descripton = descriptons[prediction_]
    return predicted_class, descripton



def prediction_main(filepath, model=model):
    img_data = [prepare_image(filepath)]
    prediction = model.predict(img_data)
    percent = round(np.max(prediction) * 100, 2)
    disease, descripton = prediction_cls(prediction)
    
    return percent, disease, descripton
