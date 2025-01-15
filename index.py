import cv2
import mediapipe as mp

# Initialize MediaPipe Hands model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Set up webcam capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()

    # Flip the frame horizontally for a mirror effect
    frame = cv2.flip(frame, 1)

    # Convert the frame to RGB (MediaPipe uses RGB images)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the image to get hand landmarks
    result = hands.process(rgb_frame)

    # Draw hand landmarks and recognize gestures
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Draw landmarks on the frame
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Gesture recognition logic (based on the position of specific landmarks)
            # Example: Check if thumb and index finger are close together (gesture "thumbs up")
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            
            # Calculate distance between thumb and index finger
            distance = ((thumb_tip.x - index_tip.x) ** 2 + (thumb_tip.y - index_tip.y) ** 2) ** 0.5

            # Gesture recognition based on distance
            if distance < 0.05:  # Adjust threshold based on your experiment
                cv2.putText(frame, "Gesture: Thumbs Up", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Additional gestures can be added here by analyzing landmarks positions

    # Display the resulting frame
    cv2.imshow('Hand Gesture Recognition', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
