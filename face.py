import cv2
import numpy as np

def get_face_detector(model_path=None, config_path=None, quantized=False):
    """
    Returns a pre-loaded face detection DNN model.
    
    Parameters:
    - model_path: Path to the model weights.
    - config_path: Path to the model configuration.
    - quantized: If True, loads the quantized TensorFlow model.
    
    Returns:
    - model: The OpenCV DNN face detector model.
    """
    if quantized:
        model_path = model_path or "models/opencv_face_detector_uint8.pb"
        config_path = config_path or "models/opencv_face_detector.pbtxt"
        model = cv2.dnn.readNetFromTensorflow(model_path, config_path)
    else:
        model_path = model_path or "models/res10_300x300_ssd_iter_140000.caffemodel"
        config_path = config_path or "models/deploy.prototxt"
        model = cv2.dnn.readNetFromCaffe(config_path, model_path)
    
    return model


def find_faces(image, model, confidence_threshold=0.5):
    """
    Detects faces in an image using a DNN model.

    Parameters:
    - image: Input image in BGR format.
    - model: Preloaded face detection DNN model.
    - confidence_threshold: Minimum confidence for valid detections.

    Returns:
    - faces: List of bounding boxes [x1, y1, x2, y2] for each detected face.
    """
    height, width = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    model.setInput(blob)
    output = model.forward()
    
    faces = []
    for i in range(output.shape[2]):
        confidence = output[0, 0, i, 2]
        if confidence > confidence_threshold:
            box = output[0, 0, i, 3:7] * np.array([width, height, width, height])
            x1, y1, x2, y2 = box.astype("int")
            faces.append([x1, y1, x2, y2])
    
    return faces


def draw_faces(image, faces, color=(0, 0, 255), thickness=3):
    """
    Draws rectangles around detected faces.

    Parameters:
    - image: Image to draw on.
    - faces: List of face bounding boxes.
    - color: Rectangle color.
    - thickness: Line thickness.
    """
    for x1, y1, x2, y2 in faces:
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)