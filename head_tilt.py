import cv2
import numpy as np
import math
from face import get_face_detector, find_faces
from facial_points import get_landmark_model, detect_marks

def get_2d_projection(img, rvec, tvec, cam_matrix, val):
    """Projects 3D box points to 2D for annotation."""
    rear_size, rear_depth, front_size, front_depth = val
    points_3d = np.array([
        (-rear_size, -rear_size, rear_depth),
        (-rear_size, rear_size, rear_depth),
        (rear_size, rear_size, rear_depth),
        (rear_size, -rear_size, rear_depth),
        (-rear_size, -rear_size, rear_depth),
        (-front_size, -front_size, front_depth),
        (-front_size, front_size, front_depth),
        (front_size, front_size, front_depth),
        (front_size, -front_size, front_depth),
        (-front_size, -front_size, front_depth)
    ], dtype=np.float32)

    dist_coeffs = np.zeros((4, 1))
    points_2d, _ = cv2.projectPoints(points_3d, rvec, tvec, cam_matrix, dist_coeffs)
    return np.int32(points_2d.reshape(-1, 2))


def draw_pose_box(img, rvec, tvec, cam_matrix, color=(255, 255, 0), thickness=2):
    """Draws the 3D head pose annotation box."""
    h, w = img.shape[:2]
    points_2d = get_2d_projection(
        img, rvec, tvec, cam_matrix, [1, 0, w, w * 2]
    )
    cv2.polylines(img, [points_2d], True, color, thickness, cv2.LINE_AA)
    for i in range(1, 4):
        cv2.line(img, tuple(points_2d[i]), tuple(points_2d[i + 5]), color, thickness, cv2.LINE_AA)

def get_head_direction_points(img, rvec, tvec, cam_matrix):
    """Returns two key 2D points to draw orientation line."""
    h, w = img.shape[:2]
    points_2d = get_2d_projection(
        img, rvec, tvec, cam_matrix, [1, 0, w, w * 2]
    )
    return points_2d[2], (points_2d[5] + points_2d[8]) // 2

def calculate_angles(p1, p2):
    """Computes angle between two points."""
    try:
        slope = (p2[1] - p1[1]) / (p2[0] - p1[0])
        return int(math.degrees(math.atan(slope)))
    except ZeroDivisionError:
        return 90

def calculate_perpendicular_angle(p1, p2):
    """Computes perpendicular angle between two points."""
    try:
        slope = (p2[1] - p1[1]) / (p2[0] - p1[0])
        return int(math.degrees(math.atan(-1 / slope)))
    except ZeroDivisionError:
        return 90

def get_camera_matrix(img_shape):
    """Returns camera matrix given image size."""
    h, w = img_shape[:2]
    focal_length = w
    center = (w / 2, h / 2)
    return np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype="double")

def classify_orientation(ang1, ang2, img, p1, x1, font):
    """Classifies and overlays head direction labels."""
    if ang1 >= 48:
        cv2.putText(img, 'Head down', (30, 30), font, 2, (255, 255, 128), 3)
    elif ang1 <= -48:
        cv2.putText(img, 'Head up', (30, 30), font, 2, (255, 255, 128), 3)

    if ang2 >= 48:
        cv2.putText(img, 'Head right', (90, 30), font, 2, (255, 255, 128), 3)
    elif ang2 <= -48:
        cv2.putText(img, 'Head left', (90, 30), font, 2, (255, 255, 128), 3)

    # Debug: Display angle values
    cv2.putText(img, str(ang1), tuple(p1), font, 2, (128, 255, 255), 3)
    cv2.putText(img, str(ang2), tuple(x1), font, 2, (255, 255, 128), 3)

# Initialization
face_model = get_face_detector()
landmark_model = get_landmark_model()
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX
ret, frame = cap.read()
camera_matrix = get_camera_matrix(frame.shape)

model_points = np.array([
    (0.0, 0.0, 0.0),             # Nose tip
    (0.0, -330.0, -65.0),        # Chin
    (-225.0, 170.0, -135.0),     # Left eye left corner
    (225.0, 170.0, -135.0),      # Right eye right corner
    (-150.0, -150.0, -125.0),    # Left mouth corner
    (150.0, -150.0, -125.0)      # Right mouth corner
], dtype='double')

# Real-time processing loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    faces = find_faces(frame, face_model)
    for face in faces:
        landmarks = detect_marks(frame, landmark_model, face)
        image_points = np.array([
            landmarks[30], landmarks[8],
            landmarks[36], landmarks[45],
            landmarks[48], landmarks[54]
        ], dtype='double')

        dist_coeffs = np.zeros((4, 1))
        success, rvec, tvec = cv2.solvePnP(
            model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_UPNP
        )

        nose_tip = tuple(map(int, image_points[0]))
        nose_direction, nose_base = cv2.projectPoints(
            np.array([(0.0, 0.0, 1000.0)]), rvec, tvec, camera_matrix, dist_coeffs
        )[0][0][0].astype(int)

        head_tip = tuple(nose_direction)
        x1, x2 = get_head_direction_points(frame, rvec, tvec, camera_matrix)

        # Draw head pose lines
        cv2.line(frame, nose_tip, head_tip, (0, 255, 255), 2)
        cv2.line(frame, tuple(x1), tuple(x2), (255, 255, 0), 2)

        ang1 = calculate_angles(nose_tip, head_tip)
        ang2 = calculate_perpendicular_angle(x1, x2)
        classify_orientation(ang1, ang2, frame, nose_tip, x1, font)

    cv2.imshow('Head Pose Estimation', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
