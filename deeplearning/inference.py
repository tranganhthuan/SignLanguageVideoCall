import cv2
import json
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model
import time

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR-CONVERSION BGR-to-RGB
    image.flags.writeable = False                  # Convert image to not-writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Convert image to writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR-COVERSION RGB-to-BGR
    return image, results

def draw_styled_landmarks(image, results):
    # Draw face connections
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, 
                             mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1), 
                             mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                             ) 
    # Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                             ) 
    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                             ) 
    # Draw right hand connections  
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                             ) 
    
def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

def preprocess_data(X):
    lh = X[:,:,:63].reshape(X.shape[0], X.shape[1], 21, 3)
    rh = X[:,:,63:].reshape(X.shape[0], X.shape[1], 21, 3)

    lh = lh - lh.mean(2)[:,:,np.newaxis,:]
    rh = rh - rh.mean(2)[:,:,np.newaxis,:]

    lh = lh.reshape(X.shape[0], X.shape[1], 63)
    rh = rh.reshape(X.shape[0], X.shape[1], 63)
    
    X_output = np.concatenate([lh, rh], axis=-1)
    return X_output

def flip_data(X):
    lh = X[:,:,:63].reshape(X.shape[0], X.shape[1], 21, 3)
    rh = X[:,:,63:].reshape(X.shape[0], X.shape[1], 21, 3)

    lh[:,:,:,0] = -lh[:,:,:,0]
    rh[:,:,:,0] = -rh[:,:,:,0]

    lh = lh.reshape(X.shape[0], X.shape[1], 63)
    rh = rh.reshape(X.shape[0], X.shape[1], 63)

    X_augment = np.concatenate([lh, rh], axis=-1)
    X_output = np.concatenate([X, X_augment], axis=0)

    return X_output

# Load the model
model_lstm = load_model('best_model.keras')
mp_holistic = mp.solutions.holistic 
mp_drawing = mp.solutions.drawing_utils 
# Load the label_to_text dictionary
with open('label_to_text.json', 'r') as f:
    label_to_text = json.load(f)

# Initialize variables
sequence = []
sentence = []
predictions = []
threshold = 0.7

cap = cv2.VideoCapture(1)
# Set mediapipe model 
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        start_time = time.time()

        # Read feed
        ret, frame = cap.read()
        if not ret:
            break

        # Make detections
        image, results = mediapipe_detection(frame, holistic)
        
        # Draw landmarks
        draw_styled_landmarks(image, results)
        
        # 2. Prediction logic
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-10:]
        
        if len(sequence) == 10:
            input_data = np.expand_dims(sequence, axis=0)[:, :, -126:]
            input_data = preprocess_data(input_data)
            input_data = flip_data(input_data)
            res = model_lstm.predict(input_data, verbose = 0).mean(axis=0)
            predictions.append(np.argmax(res))
            
            # Convert predictions to text
            # Check threshold   
            print(label_to_text[str(np.argmax(res))], res[np.argmax(res)])
            if res[np.argmax(res)] >= threshold:
                sentence.append(label_to_text[str(np.argmax(res))])
            else:
                sentence.append("None")
            sentence = sentence[-3:]

        cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
        cv2.putText(image, ' '.join(sentence), (3,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Show to screen
        cv2.imshow('Realtime LSTM Sign Language Detection', image)

        # Break gracefully
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

        # Delay to achieve 10 FPS
        time_elapsed = time.time() - start_time
        time_to_sleep = max(0.1 - time_elapsed, 0)
        time.sleep(time_to_sleep)

    cap.release()
    cv2.destroyAllWindows()