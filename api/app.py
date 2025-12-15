from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import uvicorn

app = FastAPI(title="Eye State Detection API")

# Charger le modèle Keras
MODEL_PATH = "dl-model/eye_state_model.keras"
model = tf.keras.models.load_model(MODEL_PATH)

def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("L")
    image = image.resize((64, 64))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=(0, -1))
    return image_array

@app.get("/")
def home():
    return {"message": "Eye State Detection API is running on Azure!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        input_data = preprocess_image(contents)
        prediction = model.predict(input_data)
        label = "sleepy" if prediction[0][0] < 0.5 else "awake"
        return JSONResponse({
            "prediction": label,
            "confidence": float(prediction[0][0])
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=80, reload=True)