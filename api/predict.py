from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np
import tensorflow as tf
import io

app = FastAPI()

model = tf.keras.models.load_model("saved_model_palm_disease.keras")
labels = ['Boron Excess', 'Ganoderma', 'Healthy', 'Scale insect']

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB").resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array)[0]
    class_index = int(np.argmax(preds))
    confidence = float(np.max(preds))

    return JSONResponse(content={
        "class": labels[class_index],
        "confidence": round(confidence, 4)
    })
