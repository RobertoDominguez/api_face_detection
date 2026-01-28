import os
import numpy as np
from PIL import Image
import io
from fastapi import UploadFile

BASE_DIR = "data"

def load_image(file: UploadFile):
    image_bytes = file.file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return np.array(image)

def get_faces_dir(db: str):
    return os.path.join(BASE_DIR, db, "faces")

def save_embedding(db: str, code: str, embedding: np.ndarray):
    faces_dir = get_faces_dir(db)
    user_dir = os.path.join(faces_dir, code)
    os.makedirs(user_dir, exist_ok=True)

    idx = len(os.listdir(user_dir))
    path = os.path.join(user_dir, f"{idx}.npy")
    np.save(path, embedding)

def load_all_embeddings(db: str):
    faces_dir = get_faces_dir(db)

    if not os.path.exists(faces_dir):
        return [], []

    embeddings, labels = [], []

    for code in os.listdir(faces_dir):
        code_dir = os.path.join(faces_dir, code)
        for f in os.listdir(code_dir):
            embeddings.append(np.load(os.path.join(code_dir, f)))
            labels.append(code)

    return embeddings, labels

def list_faces(db: str):
    faces_dir = get_faces_dir(db)
    result = {}

    if not os.path.exists(faces_dir):
        return result

    for code in os.listdir(faces_dir):
        count = len(os.listdir(os.path.join(faces_dir, code)))
        result[code] = count

    return result

def destroy_face(db: str, code: str):
    faces_dir = get_faces_dir(db)
    user_dir = os.path.join(faces_dir, code)

    if not os.path.exists(user_dir):
        return False

    for f in os.listdir(user_dir):
        os.remove(os.path.join(user_dir, f))
    os.rmdir(user_dir)

    return True
