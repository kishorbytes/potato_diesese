from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
from fastapi.middleware.cors import CORSMiddleware
import requests



# import os
# print(os.path.abspath("../potato_diesese/saved_models/1.keras"))

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ENDPOINT = "http://localhost:8501/v1/models/potatoes_model:predict"
ENDPOINT = "http://localhost:8001/predict" 

#this is to check the server
@app.get('/check')
async def check():
    return "Hellow Kishor?"

MODEL = tf.keras.models.load_model("../potato_diesese/saved_models/1.keras")
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]


def read_file_as_image(data)->np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post('/predict')
# async def predict(file: UploadFile = File(...)):
async def predict(file: UploadFile | None = None):
    image = read_file_as_image(await file.read()) 

    # #this is a single image we built our model in batch form (32 images), so we need to convert into bacth
    img_batch = np.expand_dims(image,axis=0) #convert 1D to 2D array
    
    # json_data = {
    #     "instances" : img_batch.tolist()
    # }
    # response = requests.post(ENDPOINT, json=json_data)
    # print(response);
    # prediction = np.array(response.json()["predictions"][0])

    # predicted_class = CLASS_NAMES[np.argmax(prediction)]
    # confidence = np.max(prediction)

    # return {
    #     "class" : predicted_class,
    #     "confidence" : confidence
    # }
    # WHEN NOT USE REQUEST MODEL
    predictions = MODEL.predict(img_batch) 

    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    return {
        "class" : predicted_class,
        "confidence" : float(confidence)
    }
        


if __name__ == "__main__":
    uvicorn.run(app, host='127.0.0.1', port=8001)

