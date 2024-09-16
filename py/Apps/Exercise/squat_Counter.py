import cv2
import mediapipe as mp
import numpy as np

# Initialize mediapipe pose detection
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Counter for squats
counter = 0
stage = None  # Keeps track of squat stage (up or down)

# Function to calculate the angle between three points (e.g., hip, knee, and ankle)
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
        
        # Get coordinates for hips, knees, and ankles (left leg in this case)
        hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, 
               landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, 
                landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
        ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, 
                 landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
        
        # Calculate the angle at the knee (hip-knee-ankle)
        angle = calculate_angle(hip, knee, ankle)
        
        # Visualize angle
        cv2.putText(image, str(int(angle)), 
                    tuple(np.multiply(knee, [640, 480]).astype(int)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Squat logic
        if angle > 160:  # When standing
            stage = "up"
        if angle < 90 and stage == "up":  # When squatting (knee angle < 90 degrees)
            stage = "down"
            counter += 1
            print(f'Squat count: {counter}')
        
    except:
        pass
    
    # Render the pose landmarks
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    
    # Display the squat count
    cv2.putText(image, 'Squat Count: ' + str(counter), (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    # Show the frame
    cv2.imshow('Squat Counter', image)

    # Exit loop if 'q' is pressed
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()