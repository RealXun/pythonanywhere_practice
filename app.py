from flask import Flask, request, jsonify, session, url_for, redirect, render_template
import joblib

# importar el formulario custom
from flower_form import....

# Importar los ficheros serializados

classifier_loaded = joblib.load("saved_models/knn_iris.pkl")
encoder_loaded = joblib.load("saved_models/iris_label_encoder.pkl")

# Creamos la función de prediccion
def make_prediction(model, encoder, sample_json):

    # recoger los valores del formulario
    SepalLengthCm = sample_json['SepalLengthCm']
    SepalWidthCm = sample_json['SepalWidthCm']
    PetalLengthCm = sample_json['PetalLengthCm']
    PetalWidthCm = sample_json['PetalWidthCm']

    # Creamos un vector de input
    flower = [[SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm]]

    # Realizamos la prediccion
    prediction_raw = model.predict(flower)

    # Convertimos los valores raw en clase
    prediction_real = encoder.inverse_transform(prediction_raw)

    return prediction_real[0]