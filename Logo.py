import requests
import sklearn
from tensorflow import keras
from keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.applications.densenet import preprocess_input
from PIL import Image
import numpy as np
import cv2
from flask import Flask, jsonify, request

def main(data):
    class_names = {
        0: 'HP',
        1: 'adidas',
        2: 'aldi',
        3: 'apple',
        4: 'becks',
        5: 'bmw',
        6: 'carlsberg',
        7: 'chimay',
        8: 'cocacola',
        9: 'corona',
        10: 'dhl',
        11: 'erdinger',
        12: 'esso',
        13: 'fedex',
        14: 'ferrari',
        15: 'ford',
        16: 'fosters',
        17: 'google',
        18: 'guinness',
        19: 'heineken',
        20: 'milka',
        21: 'nvidia',
        22: 'paulaner',
        23: 'pepsi',
        24: 'rittersport',
        25: 'shell',
        26: 'singha',
        27: 'starbucks',
        28: 'stellaartois',
        29: 'texaco',
        30: 'tsingtao',
        31: 'ups'
    }

densenet= load_model('modeldensenet.h5')
image_path = "3.jpeg"
def predict_image(image_path, model):
    # Charge l'image
    img = Image.open(image_path)
    # Redimensionne l'image en entrée du modèle
    img = img.resize((224, 224))
    # Prétraite l'image
    img = preprocess_input(np.array(img))
    # Ajoute une dimension supplémentaire pour créer un batch d'images
    img = np.expand_dims(img, axis=0)
    # Fait la prédiction avec le modèle
    preds = model.predict(img)
    # Retourne le numéro de la classe prédite
    predicted = np.argmax(preds[0])
    predicted_class = class_names[predicted]
    return predicted_class

predict_image(image_path,densenet)

if __name__ == '__main__':
    app = Flask(__name__)
    #app.debug = True
    @app.route('/', methods=["GET", "POST"])
    def tryout():
        data = request.get_json()
        return jsonify({
            'logoPrediction': main(data)
        })
    app.run(host='0.0.0.0')