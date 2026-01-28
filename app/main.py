from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from typing import List
import face_recognition
import numpy as np
import os

from app.utils import (
    load_image,
    save_embedding,
    load_all_embeddings,
    list_faces,
    destroy_face
)
from app.spoofing import is_real_face

app = FastAPI()

@app.post("/register")
async def register_face(
    db: str = Form(...),
    code: str = Form(...),
    images: List[UploadFile] = File(...)
):
    if len(images) < 3:
        raise HTTPException(400, "Mínimo 3 imágenes")

    saved = 0

    for img in images:
        image = load_image(img)

        if not is_real_face(image):
            continue

        locations = face_recognition.face_locations(image)
        encodings = face_recognition.face_encodings(image, locations)

        if len(encodings) == 1:
            save_embedding(db, code, encodings[0])
            saved += 1

    if saved == 0:
        raise HTTPException(400, "No se pudieron registrar caras válidas")

    return {
        "db": db,
        "code": code,
        "registered": saved
    }

@app.post("/recognize")
async def recognize_face(
    db: str = Form(...),
    images: List[UploadFile] = File(...)
):
    if not images:
        raise HTTPException(400, "Debe enviar al menos una imagen")

    collected_embeddings = []

    for img in images:
        image_np = load_image(img)

        if not is_real_face(image_np):
            continue

        encodings = face_recognition.face_encodings(image_np)

        if len(encodings) == 1:
            collected_embeddings.append(encodings[0])

    if not collected_embeddings:
        raise HTTPException(403, "No se obtuvieron caras válidas")

    mean_embedding = np.mean(collected_embeddings, axis=0)

    known_embeddings, labels = load_all_embeddings(db)
    if not known_embeddings:
        raise HTTPException(500, "No hay datos registrados en esta DB")

    distances = face_recognition.face_distance(
        known_embeddings,
        mean_embedding
    )

    best_idx = int(np.argmin(distances))
    best_distance = float(distances[best_idx])

    THRESHOLD = 0.6

    if best_distance > THRESHOLD:
        raise HTTPException(404, "Persona no reconocida")

    confidence = max(0.0, 1 - (best_distance / THRESHOLD))

    return {
        "db": db,
        "code": labels[best_idx],
        "confidence": round(confidence, 4),
        "distance": round(best_distance, 4),
        "images_used": len(collected_embeddings)
    }

@app.get("/faces")
async def list_faces_db(db: str):
    return {
        "db": db,
        "faces": list_faces(db)
    }

@app.delete("/faces/{code}")
async def delete_face(
    db: str,
    code: str
):
    deleted = destroy_face(db, code)

    if not deleted:
        raise HTTPException(404, "Persona no encontrada")

    return {
        "db": db,
        "deleted": code
    }
