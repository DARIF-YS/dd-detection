import gradio as gr
import cv2
import numpy as np
import time
import requests
from PIL import Image
import io

# Configuration
ALERT_THRESHOLD = 3  # secondes avant alerte
API_URL = "/predict" # Public URL de l’API FastAPI

# Haar cascades pour la détection des yeux
left_eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_lefteye_2splits.xml")
right_eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_righteye_2splits.xml")

closed_start_time = None

def predict_eye_via_api(gray_eye):
    """Envoie l'image d'un œil à l'API FastAPI et récupère la prédiction"""
    try:
        # Convertir en image PNG
        img_pil = Image.fromarray(gray_eye)
        buf = io.BytesIO()
        img_pil.save(buf, format="PNG")
        buf.seek(0)

        # Requête POST à l’API
        response = requests.post(API_URL, files={"file": ("eye.png", buf, "image/png")})
        if response.status_code == 200:
            data = response.json()
            return "Closed" if data["prediction"] == "sleepy" else "Open"
        else:
            print("Erreur API:", response.text)
            return "Open"
    except Exception as e:
        print("Erreur lors de la requête API:", e)
        return "Open"

def detect_and_draw(cascade, gray, frame):
    eyes = cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=7, minSize=(30, 30))
    if len(eyes) == 0:
        return 1  # pas d’œil détecté → considéré fermé
    
    x, y, w, h = eyes[0]
    roi_gray = gray[y:y+h, x:x+w]
    state = predict_eye_via_api(roi_gray)
    
    color = (0, 255, 0) if state == "Open" else (255, 0, 0)
    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
    cv2.putText(frame, state, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    return 1 if state == "Closed" else 0

def process_frame(input_image):
    global closed_start_time

    if input_image is None:
        return None

    frame = np.array(input_image)
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    eyes_closed_count = detect_and_draw(left_eye_cascade, gray, frame) + \
                        detect_and_draw(right_eye_cascade, gray, frame)

    if eyes_closed_count >= 2:
        if closed_start_time is None:
            closed_start_time = time.time()
        elapsed = time.time() - closed_start_time
        progress = min(elapsed / ALERT_THRESHOLD, 1)
        cv2.rectangle(frame, (50, 50), (350, 80), (255, 255, 255), 2)
        cv2.rectangle(frame, (50, 50), (50 + int(progress * 300), 80), (255, 0, 0), -1)
        text = f"Eyes closed: {elapsed:.1f}s"
        cv2.putText(frame, text, (50, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        if elapsed >= ALERT_THRESHOLD:
            cv2.putText(frame, "DROWSINESS ALERT!", (80, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)
    else:
        closed_start_time = None
        cv2.rectangle(frame, (50, 50), (350, 80), (255, 255, 255), 2)
        cv2.putText(frame, "Eyes open", (50, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    return frame

# Interface Gradio
with gr.Interface(
    fn=process_frame,
    inputs=gr.Image(sources=["webcam"], streaming=True),
    outputs="image",
    live=True,
    title="Détection de Somnolence (API Azure)",
    description="Regardez la caméra. Les frames sont envoyées à l’API pour prédire si vos yeux sont ouverts ou fermés."
) as demo:
    demo.launch()
