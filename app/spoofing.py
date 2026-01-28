import cv2
import numpy as np

def is_real_face(image: np.ndarray) -> bool:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # 1. Varianza Laplaciana (textura)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if laplacian_var < 30:
        return False

    # 2. Detección de bordes
    edges = cv2.Canny(gray, 50, 150)
    edge_density = edges.mean()

    if edge_density < 5:
        return False

    return True
