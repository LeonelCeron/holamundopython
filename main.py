from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Nombre(BaseModel):
    nombre: str
    apellido: str
    edad: int

@app.get("/inicio")
async def ruta_de_prueba():
    return {"Message": "Hola grupo 08 desde Fast_API v1"}


