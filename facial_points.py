import cv2
import numpy as np
import tensorflow as tf


def get_landmark_model(model_path='models/pose_model'):
    """
    Load and return the facial landmark TensorFlow model.
    
    Parameters:
    - model_path: Path to the saved model directory.
    
    Returns:
    - Loaded TensorFlow model.
    """
    return tf.saved_model.load(model_path)


def get_square_box(box):
    """
    Convert a rectangular bounding box to a square one by expanding the shorter side.
    
    Parameters:
    - box: List of [x1, y1, x2, y2]
    
    Returns:
    - Square bounding box [x1, y1, x2, y2]
    """
    x1, y1, x2, y2 = box
    width = x2 - x1
    height = y2 - y1
    diff = height - width
    delta = abs(diff) // 2

    if diff > 0:
        x1 -= delta
        x2 += delta + (diff % 2)
    elif diff < 0:
        y1 -= delta
        y2 += delta + (abs(diff) % 2)

    assert (x2 - x1) == (y2 - y1), 'Box is not square.'

    return [x1, y1, x2, y2]


def move_box(box, offset):
    """
    Move a box by a given offset.
    
    Parameters:
    - box: List of [x1, y1, x2, y2]
    - offset: List of [dx, dy]
    
    Returns:
    - Moved box [x1+dx, y1+dy, x2+dx, y2+dy]
    """
    dx, dy = offset
    x1, y1, x2, y2 = box
    return [x1 + dx, y1 + dy, x2 + dx, y2 + dy]


def detect_marks(image, model, face_box):
    """
    Detect facial landmarks using a pre-trained model.
    
    Parameters:
    - image: Input BGR image
    - model: Loaded TensorFlow facial landmark model
    - face_box: List [x1, y1, x2, y2] of face bounding box
    
    Returns:
    - marks: np.ndarray of shape (68, 2) with landmark positions
    """
    offset_y = int(0.1 * abs(face_box[3] - face_box[1]))
    shifted_box = move_box(face_box, [0, offset_y])
    square_box = get_square_box(shifted_box)

    h, w = image.shape[:2]
    x1, y1, x2, y2 = map(int, square_box)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    face_crop = image[y1:y2, x1:x2]
    face_resized = cv2.resize(face_crop, (128, 128))
    face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)

    # Run inference
    result = model.signatures["predict"](tf.constant([face_rgb], dtype=tf.uint8))
    landmarks = np.array(result['output']).flatten()[:136].reshape(-1, 2)

    landmarks *= (x2 - x1)
    landmarks[:, 0] += x1
    landmarks[:, 1] += y1

    return landmarks.astype(np.uint)


def draw_marks(image, landmarks, color=(0, 255, 0)):
    """
    Draw facial landmarks on the image.
    
    Parameters:
    - image: Image on which landmarks are drawn
    - landmarks: np.ndarray of shape (N, 2)
    - color: Landmark point color
    """
    for x, y in landmarks:
        cv2.circle(image, (x, y), 2, color, -1, cv2.LINE_AA)
