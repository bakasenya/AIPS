import cv2
from face import get_face_detector, find_faces
from facial_points import get_landmark_model, detect_marks, draw_marks

# Load models
face_model = get_face_detector()
landmark_model = get_landmark_model()

# Points to track for mouth (outer and inner)
outer_points = [[49, 59], [50, 58], [51, 57], [52, 56], [53, 55]]
inner_points = [[61, 67], [62, 66], [63, 65]]

# Initialize variables
d_outer = [0] * len(outer_points)
d_inner = [0] * len(inner_points)
font = cv2.FONT_HERSHEY_SIMPLEX 

# Initialize video capture
cap = cv2.VideoCapture(0)

def calculate_mouth_distances(shape):
    """Calculate the mouth distances to use for comparison."""
    global d_outer, d_inner
    for i, (p1, p2) in enumerate(outer_points):
        d_outer[i] += shape[p2][1] - shape[p1][1]
    for i, (p1, p2) in enumerate(inner_points):
        d_inner[i] += shape[p2][1] - shape[p1][1]

def normalize_distances():
    """Normalize the calculated distances by dividing by 100."""
    global d_outer, d_inner
    d_outer = [x / 100 for x in d_outer]
    d_inner = [x / 100 for x in d_inner]

def detect_mouth_opening(shape):
    """Detect if the mouth is open based on distance thresholds."""
    cnt_outer, cnt_inner = 0, 0
    for i, (p1, p2) in enumerate(outer_points):
        if d_outer[i] + 3 < shape[p2][1] - shape[p1][1]:
            cnt_outer += 1 
    for i, (p1, p2) in enumerate(inner_points):
        if d_inner[i] + 2 < shape[p2][1] - shape[p1][1]:
            cnt_inner += 1
    return cnt_outer > 3 and cnt_inner > 2

def main():
    # Initial capturing phase to record mouth distances
    while True:
        ret, img = cap.read()
        if not ret:
            break
        
        rects = find_faces(img, face_model)
        for rect in rects:
            shape = detect_marks(img, landmark_model, rect)
            draw_marks(img, shape)
            cv2.putText(img, 'Press r to record Mouth distances', (30, 30), font, 1, (0, 255, 255), 2)
            cv2.imshow("Output", img)
        
        if cv2.waitKey(1) & 0xFF == ord('r'):
            for _ in range(100):  # Record for 100 frames
                for rect in rects:
                    shape = detect_marks(img, landmark_model, rect)
                    calculate_mouth_distances(shape)
            break

    normalize_distances()
    cv2.destroyAllWindows()

    # Main loop for mouth detection
    while True:
        ret, img = cap.read()
        if not ret:
            break
        rects = find_faces(img, face_model)
        for rect in rects:
            shape = detect_marks(img, landmark_model, rect)
            draw_marks(img, shape[48:])
            
            if detect_mouth_opening(shape):
                print('Mouth open')
                cv2.putText(img, 'Mouth open', (30, 30), font, 1, (0, 255, 255), 2)
        
        cv2.imshow("Output", img)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

