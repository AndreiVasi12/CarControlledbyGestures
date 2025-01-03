import mediapipe as mp
import cv2
import numpy as np
import joblib
import RPi.GPIO as GPIO

# Load the model
model_name_rf = 'random_forest_dt_2024_05_16__20_26_41__acc_1.0.pkl'
model = joblib.load(model_name_rf)

# Mediapipe setup
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Mapping of class indices to labels and actions
idx_to_class = {
    0: 'hand_closed',
    1: 'one',
    2: 'two',
    4: 'palm',
    5: 'hand_open',
}

# Define GPIO pins for motor control
motor1_pin1 = 17
motor1_pin2 = 27
motor2_pin1 = 23
motor2_pin2 = 24

# Initialize GPIO pins
def initialize_GPIO():
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(motor1_pin1, GPIO.OUT)
    GPIO.setup(motor1_pin2, GPIO.OUT)
    GPIO.setup(motor2_pin1, GPIO.OUT)
    GPIO.setup(motor2_pin2, GPIO.OUT)

#GPIO Car controlling fuctions for command
def cleanup_GPIO():
    GPIO.cleanup()

def stop_motors():
    GPIO.output(motor1_pin1, GPIO.LOW)
    GPIO.output(motor1_pin2, GPIO.LOW)
    GPIO.output(motor2_pin1, GPIO.LOW)
    GPIO.output(motor2_pin2, GPIO.LOW)

def move_forward():
    GPIO.output(motor1_pin1, GPIO.HIGH)
    GPIO.output(motor1_pin2, GPIO.LOW)
    GPIO.output(motor2_pin1, GPIO.HIGH)
    GPIO.output(motor2_pin2, GPIO.LOW)

def move_backward():
    GPIO.output(motor1_pin1, GPIO.LOW)
    GPIO.output(motor1_pin2, GPIO.HIGH)
    GPIO.output(motor2_pin1, GPIO.LOW)
    GPIO.output(motor2_pin2, GPIO.HIGH)

def turn_left():
    GPIO.output(motor1_pin1, GPIO.LOW)
    GPIO.output(motor1_pin2, GPIO.HIGH)
    GPIO.output(motor2_pin1, GPIO.HIGH)
    GPIO.output(motor2_pin2, GPIO.LOW)

def turn_right():
    GPIO.output(motor1_pin1, GPIO.HIGH)
    GPIO.output(motor1_pin2, GPIO.LOW)
    GPIO.output(motor2_pin1, GPIO.LOW)
    GPIO.output(motor2_pin2, GPIO.HIGH)

# Initialize variables
current_command = None
#intialize mediapipe hand detection and opening camera
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.4)
capture = cv2.VideoCapture(0)  # 0 for integrated camera, 1 for plugged camera

initialize_GPIO()  # Initialize GPIO pins

try:
    while capture.isOpened():
        ret, frame = capture.read()
        height, width = frame.shape[:-1]
        frame = cv2.flip(frame, 1)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detected_image = hands.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        x = []

        if detected_image.multi_hand_landmarks:
            for hand_lms in detected_image.multi_hand_landmarks:
                for lm in hand_lms.landmark:
                    x.extend([lm.x, lm.y])  # Extract x, y coordinates of each landmark

                # Draw hand landmarks on the image
                mp_drawing.draw_landmarks(image, hand_lms, mp_hands.HAND_CONNECTIONS,
                                          landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
                                              color=(255, 0, 255), thickness=4, circle_radius=2),
                                          connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
                                              color=(20, 180, 90), thickness=2, circle_radius=2)
                                          )

        # If hand landmarks are detected
        if x:
            x = np.array(x)  # Convert to numpy array
            x = x.reshape(1, -1)  # Reshape to (1, 42) if are more than 42 labels
            if x.shape[1] == 42:
                yhat_idx = int(model.predict(x)[0])  # Make prediction based on model
                yhat = idx_to_class[yhat_idx]  # Get predicted label

                # Display prediction on the image
                cv2.putText(image, f'{yhat}', (50, 50), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 0, 0), 2)

                # Perform the action associated with the predicted label
                if current_command != yhat:
                    if yhat == 'one':
                        move_forward()
                    elif yhat == 'two':
                        move_backward()
                    elif yhat == 'palm':  # Using 'palm' to indicate a left turn
                        turn_left()
                    elif yhat == 'hand_open':  # Using 'hand_open' to indicate a right turn
                        turn_right()
                    else:
                        stop_motors()
                    current_command = yhat
            else:
                stop_motors()
        else:
            stop_motors()

        # Display the image with hand landmarks
        cv2.imshow('Hand Gesture Reader', image)

        # Check for 'q' key to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Release resources
    capture.release()
    cv2.destroyAllWindows()
    cleanup_GPIO()  # Cleanup GPIO pins
