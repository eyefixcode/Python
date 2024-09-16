import cv2
import mediapipe as mp
import numpy as np

# Initialize mediapipe pose detection
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Counter for sit-ups
counter = 0
stage = None  # Keeps track of up/down stage

# Function to calculate the Euclidean distance between two points
def calculate_distance(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.linalg.norm(a - b)

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
        
        # Get coordinates for head (nose) and knees
        head = [landmarks[mp_pose.PoseLandmark.NOSE.value].x, 
                landmarks[mp_pose.PoseLandmark.NOSE.value].y]
        knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, 
                landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
        
        # Calculate distance between head and knee
        distance = calculate_distance(head, knee)
        
        # Visualize distance
        cv2.putText(image, f'Distance: {int(distance*100)}', 
                    tuple(np.multiply(knee, [640, 480]).astype(int)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Sit-up logic: Adjust the threshold to be more generous
        # Increase the threshold to make it more forgiving for counting sit-ups
        if distance > 0.5:  # Body flat (distance relatively large)
            stage = "down"
        if distance < 0.35 and stage == "down":  # Head gets "close enough" to knees
            stage = "up"
            counter += 1
            print(f'Sit-up count: {counter}')
        
    except:
        pass
    
    # Render the pose landmarks
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    
    # Display the sit-up count
    cv2.putText(image, 'Sit-up Count: ' + str(counter), (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    # Show the frame
    cv2.imshow('Sit-up Counter (Side View)', image)

    # Exit loop if 'q' is pressed
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()