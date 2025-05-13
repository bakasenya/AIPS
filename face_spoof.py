import numpy as np
import cv2
import joblib
from face import get_face_detector, find_faces

def compute_normalized_histogram(img):
    hist_channels = []
    for channel in range(3):
        hist = cv2.calcHist([img], [channel], None, [256], [0, 256])
        hist *= 255.0 / hist.max()
        hist_channels.append(hist)
    return np.array(hist_channels)

# Load face detector and trained spoofing classifier
face_detector = get_face_detector()
spoof_model = joblib.load('models/face_spoofing.pkl')

cap = cv2.VideoCapture(0)
# Spoof detection sampling settings
sample_window_size = 1
frame_count = 0
probability_samples = np.zeros(sample_window_size, dtype=np.float32)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    detected_faces = find_faces(frame, face_detector)
    probability_samples[frame_count % sample_window_size] = 0
    height, width = frame.shape[:2]

    for x1, y1, x2, y2 in detected_faces:
        face_roi = frame[y1:y2, x1:x2]
        label_position = (x1, y1 - 5)

        # Color space conversions
        face_ycrcb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2YCR_CB)
        face_luv = cv2.cvtColor(face_roi, cv2.COLOR_BGR2LUV)

        # Compute histograms
        hist_ycrcb = compute_normalized_histogram(face_ycrcb)
        hist_luv = compute_normalized_histogram(face_luv)

        # Create feature vector and reshape
        feature_vector = np.append(hist_ycrcb.ravel(), hist_luv.ravel()).reshape(1, -1)

        # Predict spoof probability
        spoof_prob = spoof_model.predict_proba(feature_vector)[0][1]
        probability_samples[frame_count % sample_window_size] = spoof_prob

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # Display label based on average probability
        if 0 not in probability_samples:
            label = "False" if np.mean(probability_samples) >= 0.7 else "True"
            label_color = (0, 0, 255) if label == "False" else (0, 255, 0)

            cv2.putText(
                img=frame,
                text=label,
                org=label_position,
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.9,
                color=label_color,
                thickness=2,
                lineType=cv2.LINE_AA
            )

    frame_count += 1
    cv2.imshow('Face Spoof Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
