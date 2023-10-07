from fastapi import FastAPI
from pydantic import BaseModel


import csv
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import pickle
import matplotlib.pyplot as plt
import seaborn as sb


app = FastAPI()

class Nombre(BaseModel):
    nombre: str
    apellido: str
    edad: int

#names = ['prenatalidad','glucosa','presion_sangre','piel','insulina','imc','antecedente','edad','outcome']
class DatosPaciente(BaseModel):
    prenatalidad: float
    glucosa: float
    presion_sangre: float
    piel: float
    insulina: float
    imc: float
    antecedente: float
    edad: float

@app.get("/inicio")
async def ruta_de_prueba():
    return {"Message": "Hola grupo 08 desde Fast_API v1"}


@app.post("/insertar")
async def insertar_prueba(nombre: Nombre):
    return {"Message": "Los datos del usuario son los siguientes","nombre":{nombre.nombre},"apellido":{nombre.apellido},"edad":{nombre.edad}}

def hola():
    
    return {"Hola"}




@app.post("/consultar")
async def consultar_paciente(datosPaciente: DatosPaciente):
    #aux = hola()
    names = ['PRENATALIDAD','GLUCOSA','PRESION_SANGRE','PIEL','INSULINA','IMC','ANTECEDENTE','EDAD','Outcome']

    filename = 'content/pima-indians-diabetes.csv'
    raw_data = open(filename,'r')
    leerData = csv.reader(raw_data, delimiter=',', quoting=csv.QUOTE_NONE)
    x = list(leerData)
    data = np.array(x)
    print(data.shape)

    #############

    dataPd = pd.read_csv(filename, names=names)

    dataPd['Outcome'].value_counts()
    x = dataPd.drop(columns='Outcome', axis=1)
    y = dataPd['Outcome']


    x_train, x_test, y_train, y_test=train_test_split(x,y, test_size=0.2)
    #print(x.shape, x_train.shape, x_test.shape)
    #Definiendo el modelo a usar
    clasifier = svm.SVC(kernel='linear')
    clasifier.fit(x_train, y_train)
    input_data = (
        datosPaciente.prenatalidad,
        datosPaciente.glucosa,
        datosPaciente.presion_sangre,
        datosPaciente.piel,
        datosPaciente.insulina,
        datosPaciente.imc,
        datosPaciente.antecedente,
        datosPaciente.edad
    )
    #input_data = (5,166,72,19,175,25.8,0.587,51) Sí es
    #input_data = (5,80,72,19,90,25.8,0.587,51)  No es
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_ajustado = input_data_as_numpy_array.reshape(1,-1) 
    prediction =clasifier.predict(input_data_ajustado)
    #print(prediction)

    if(prediction[0] == 0 ):
        resultado = 'La persona no es potencialmente diabetica'
    else:
        resultado = 'La persona es potencialmente diabetica'
    
    return { "Message": resultado}






