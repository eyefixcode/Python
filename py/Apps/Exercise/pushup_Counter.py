import cv2
import mediapipe as mp
import numpy as np

# Initialize mediapipe pose detection
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Counter for push-ups
counter = 0
stage = None  # Keeps track of up/down stage

# Function to calculate the angle between three points (e.g., shoulder, elbow, and wrist)
def calculate_angle(a, b, c):
    a = np.array(a)  # First point
    b = np.array(b)  # Second point
    c = np.array(c)  # Third point
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

# Open webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    
    # Recolor image to RGB for mediapipe
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    
    # Make detections
    results = pose.process(image)
    
    # Recolor back to BGR for OpenCV
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Extract landmarks
    try:
        landmarks = results.pose_landmarks.landmark
        
        # Get coordinates for shoulders, elbows, and wrists
        shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, 
                    landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, 
                 landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, 
                 landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
        
        # Calculate angle
        angle = calculate_angle(shoulder, elbow, wrist)
        
        # Visualize angle
        cv2.putText(image, str(angle), 
                    tuple(np.multiply(elbow, [640, 480]).astype(int)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA
                    )
        
        # Push-up logic
        if angle > 160:
            stage = "up"
        if angle < 90 and stage == "up":
            stage = "down"
            counter += 1
            print(f'Push-up count: {counter}')
        
    except:
        pass
    
    # Render the pose landmarks
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    
    # Display the resulting frame
    cv2.putText(image, 'Push-up Count: ' + str(counter), (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow('Push-up Counter', image)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()