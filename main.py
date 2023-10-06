from fastapi import FastAPI

app = FastAPI()

@app.get("/inicio")
async def ruta_de_prueba():
    return {"Message": "Hola grupo 08 desde Fast_API v1"}