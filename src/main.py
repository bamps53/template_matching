import base64
from fastapi import FastAPI, UploadFile, File
import cv2
import numpy as np
from PIL import Image
import io

app = FastAPI()

template = None


@app.post("/upload_template/")
async def upload_template(file: UploadFile = File(...)):
    global template
    image_stream = io.BytesIO(await file.read())
    image_stream.seek(0)
    pil_image = Image.open(image_stream)
    template = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    return {"status": "Template uploaded successfully"}


@app.post("/upload_image/")
async def upload_image(file: UploadFile = File(...)):
    global template
    image_stream = io.BytesIO(await file.read())
    image_stream.seek(0)
    pil_image = Image.open(image_stream)
    cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    # Perform template matching
    res = cv2.matchTemplate(cv_image, template, cv2.TM_CCOEFF_NORMED)

    # Draw rectangle around the matched region
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    w, h = template.shape[::-1]
    cv2.rectangle(cv_image, max_loc, (max_loc[0] + w, max_loc[1] + h), 255, 2)

    # Convert the image back to PIL format to send as response
    pil_image = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
    byte_arr = io.BytesIO()
    pil_image.save(byte_arr, format='JPEG')
    encoded_image = base64.encodebytes(byte_arr.getvalue()).decode('ascii')

    return {"image": encoded_image, "max_val": max_val, "max_loc": max_loc}
