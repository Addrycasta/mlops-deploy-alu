from flask import Flask,request,jsonify
from flask_basicauth import BasicAuth
import os
from googletrans import Translator
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from textblob import TextBlob

modelo=pickle.load(open('modelo.pkl','rb')) # Instanciamos el modelo luego de obtener el archvivo pkl.
columnas = ['area', 'modelo','estacionamiento']

# Inicialización de la aplicación Flask
app = Flask(__name__) # Se usa name para que Flask entienda desde dónde se está ejecutando la aplicación.
app.config['BASIC_AUTH_USERNAME'] = os.environ.get('BASIC_AUTH_USERNAME')
app.config['BASIC_AUTH_PASSWORD'] = os.environ.get('BASIC_AUTH_PASSWORD')
basic_auth = BasicAuth(app)

@app.route('/')

def home():
    return 'Sincrinización exitosa'

@app.route('/sentimiento/<frase>') #Cualquier texto que pongas en esa parte de la URL <frase> será capturado y pasado como un argumento a la función sentimiento.
@basic_auth.required
def sentimiento(frase): # Definimos una nueva funció que se llama sentimiento y recive como parametro frase.
    translator = Translator()
    traduccion = translator.translate(frase, src='es', dest='en')  # Traduce la frase
    tb = TextBlob(traduccion.text)  # Crea el objeto TextBlob con el texto traducido
    polaridad = tb.sentiment.polarity  # Obtiene la polaridad del sentimiento
    return f'La polaridad de la frase es: {polaridad}' # Retornamos la valiable polaridad a travez de un string.


@app.route('/precio_casas/', methods=['POST'])
@basic_auth.required
def precio_casas():
    datos=request.get_json() # Obtenemos nuestro json
    datos_input=[datos[col]for col in columnas] # Datos de entrada en una compresion de listas, "col" variable temporal que toma cada valor de la lista columnas.
    precio = modelo.predict([datos_input]) # Como ya tenemos la lista en ciclo for, ya solo ponemos un juego de corchete. 
    return jsonify(precio= f'El precio estimado de la casa es: {precio[0]}')

# Ejecuta la aplicación
app.run(debug=True)
