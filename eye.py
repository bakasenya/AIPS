import cv2
import numpy as np
from face import get_face_detector, find_faces
from facial_points import get_landmark_model, detect_marks


# Eye indices from dlib shape
LEFT_EYE_INDICES = [36, 37, 38, 39, 40, 41]
RIGHT_EYE_INDICES = [42, 43, 44, 45, 46, 47]

def apply_eye_mask(mask, eye_indices, landmarks):
    eye_points = np.array([landmarks[i] for i in eye_indices], dtype=np.int32)
    mask = cv2.fillConvexPoly(mask, eye_points, 255)
    left = eye_points[0][0]
    top = (eye_points[1][1] + eye_points[2][1]) // 2
    right = eye_points[3][0]
    bottom = (eye_points[4][1] + eye_points[5][1]) // 2
    return mask, [left, top, right, bottom]

def determine_eyeball_position(bounds, cx, cy):
    ratio_x = (bounds[0] - cx) / (cx - bounds[2])
    ratio_y = (cy - bounds[1]) / (bounds[3] - cy)

    if ratio_x > 3:
        return 1  # Looking left
    elif ratio_x < 0.33:
        return 2  # Looking right
    elif ratio_y < 0.33:
        return 3  # Looking up
    else:
        return 0  # Looking straight

def extract_eye_contour(thresh_img, mid_x, frame, eye_bounds, is_right_eye=False):
    contours, _ = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    try:
        max_contour = max(contours, key=cv2.contourArea)
        moments = cv2.moments(max_contour)
        cx = int(moments['m10'] / moments['m00'])
        cy = int(moments['m01'] / moments['m00'])

        if is_right_eye:
            cx += mid_x

        cv2.circle(frame, (cx, cy), 4, (0, 0, 255), 2)
        position = determine_eyeball_position(eye_bounds, cx, cy)
        return position
    except:
        return None

def preprocess_threshold_image(thresh_img):
    thresh_img = cv2.erode(thresh_img, None, iterations=2)
    thresh_img = cv2.dilate(thresh_img, None, iterations=4)
    thresh_img = cv2.medianBlur(thresh_img, 3)
    thresh_img = cv2.bitwise_not(thresh_img)
    return thresh_img

def display_gaze_direction(frame, left_pos, right_pos):
    if left_pos == right_pos and left_pos != 0:
        label = ''
        if left_pos == 1:
            label = 'Looking left'
        elif left_pos == 2:
            label = 'Looking right'
        elif left_pos == 3:
            label = 'Looking up'
        print(label)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, label, (30, 30), font, 1, (0, 255, 255), 2, cv2.LINE_AA)

# Load models
face_detector = get_face_detector()
landmark_detector = get_landmark_model()

# Video capture
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
thresh_img = frame.copy()

cv2.namedWindow('image')
dilate_kernel = np.ones((9, 9), np.uint8)

cv2.createTrackbar('threshold', 'image', 75, 255, lambda x: None)

while True:
    ret, frame = cap.read()
    detected_faces = find_faces(frame, face_detector)

    for face in detected_faces:
        landmarks = detect_marks(frame, landmark_detector, face)
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)

        mask, left_eye_bounds = apply_eye_mask(mask, LEFT_EYE_INDICES, landmarks)
        mask, right_eye_bounds = apply_eye_mask(mask, RIGHT_EYE_INDICES, landmarks)
        mask = cv2.dilate(mask, dilate_kernel, 5)

        eye_region = cv2.bitwise_and(frame, frame, mask=mask)
        masked_area = (eye_region == [0, 0, 0]).all(axis=2)
        eye_region[masked_area] = [255, 255, 255]

        mid_x = (landmarks[42][0] + landmarks[39][0]) // 2
        eye_gray = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
        threshold_value = cv2.getTrackbarPos('threshold', 'image')
        _, thresh_img = cv2.threshold(eye_gray, threshold_value, 255, cv2.THRESH_BINARY)
        thresh_img = preprocess_threshold_image(thresh_img)

        left_eye_position = extract_eye_contour(thresh_img[:, 0:mid_x], mid_x, frame, left_eye_bounds)
        right_eye_position = extract_eye_contour(thresh_img[:, mid_x:], mid_x, frame, right_eye_bounds, is_right_eye=True)

        display_gaze_direction(frame, left_eye_position, right_eye_position)

    cv2.imshow('eyes', frame)
    cv2.imshow("image", thresh_img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
