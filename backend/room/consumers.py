
import json
import logging
import base64
import cv2
import os
import uuid
import numpy as np
import mediapipe as mp
from channels.generic.websocket import AsyncWebsocketConsumer
from tensorflow.keras.models import load_model

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR-CONVERSION BGR-to-RGB
    image.flags.writeable = False                  # Convert image to not-writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Convert image to writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR-COVERSION RGB-to-BGR
    return image, results

def extract_keypoints_from_json(data):
    
    pose = np.array([[lm['x'], lm['y'], lm['z'], 1.0] for lm in data['pose_landmarks']]).flatten() if data['pose_landmarks'] else np.zeros(33*4)
    face = np.array([[lm['x'], lm['y'], lm['z']] for lm in data['face_landmarks']]).flatten() if data['face_landmarks'] else np.zeros(468*3)
    lh = np.array([[lm['x'], lm['y'], lm['z']] for lm in data['left_hand_landmarks']]).flatten() if data['left_hand_landmarks'] else np.zeros(21*3)
    rh = np.array([[lm['x'], lm['y'], lm['z']] for lm in data['right_hand_landmarks']]).flatten() if data['right_hand_landmarks'] else np.zeros(21*3)
    
    return np.concatenate([pose, face, lh, rh])

def extract_keypoints_from_mediapipe(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

def output_to_json(results):
    face_landmarks = []
    if results.face_landmarks:
        for landmark in results.face_landmarks.landmark:
            face_landmarks.append({
                'x': landmark.x,
                'y': landmark.y,
                'z': landmark.z
            })

    
    pose_landmarks = []
    if results.pose_landmarks:
        for landmark in results.pose_landmarks.landmark:
            pose_landmarks.append({
                'x': landmark.x,
                'y': landmark.y,
                'z': landmark.z
            })

    left_hand_landmarks = []
    if results.left_hand_landmarks:
        for landmark in results.left_hand_landmarks.landmark:
            left_hand_landmarks.append({
                'x': landmark.x,
                'y': landmark.y,
                'z': landmark.z
            })

    
    right_hand_landmarks = []
    if results.right_hand_landmarks:
        for landmark in results.right_hand_landmarks.landmark:
            right_hand_landmarks.append({
                'x': landmark.x,
                'y': landmark.y,
                'z': landmark.z
            })
    return json.dumps({
                'face_landmarks': face_landmarks,
                'pose_landmarks': pose_landmarks,
                'left_hand_landmarks': left_hand_landmarks,
                'right_hand_landmarks': right_hand_landmarks,
            })

def numpy_to_landmarks(keypoints):
    # Define the sizes of each component
    pose_size = 33 * 4
    face_size = 468 * 3
    hand_size = 21 * 3

    # Split the keypoints array into its components
    pose = keypoints[:pose_size]
    face = keypoints[pose_size:pose_size + face_size]
    lh = keypoints[pose_size + face_size:pose_size + face_size + hand_size]
    rh = keypoints[pose_size + face_size + hand_size:]

    # Convert pose landmarks
    pose_landmarks = [{'x': x, 'y': y, 'z': z, 'visibility': v} for x, y, z, v in pose.reshape(-1, 4)] if np.any(pose) else []

    # Convert face landmarks
    face_landmarks = [{'x': x, 'y': y, 'z': z} for x, y, z in face.reshape(-1, 3)] if np.any(face) else []

    # Convert left hand landmarks
    left_hand_landmarks = [{'x': x, 'y': y, 'z': z} for x, y, z in lh.reshape(-1, 3)] if np.any(lh) else []

    # Convert right hand landmarks
    right_hand_landmarks = [{'x': x, 'y': y, 'z': z} for x, y, z in rh.reshape(-1, 3)] if np.any(rh) else []

    # Construct the final dictionary
    return {
        'pose_landmarks': pose_landmarks,
        'face_landmarks': face_landmarks,
        'left_hand_landmarks': left_hand_landmarks,
        'right_hand_landmarks': right_hand_landmarks
    }

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

class VideoConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        self.room_name = self.scope['url_route']['kwargs']['room_name']
        self.room_group_name = f'chat_{self.room_name}'

        self.pose_meta = {}
        with open('../deeplearning/highest_prob_samples.json', 'r') as f:
            self.pose_meta = json.load(f)

        # Join room group
        await self.channel_layer.group_add(
            self.room_group_name,
            self.channel_name
        )
        await self.accept()
        
        await self.send(text_data=json.dumps({
            'type': 'connection_status',
            'status': 'connected'
        }))

        await self.send_poses(None)

    async def disconnect(self, close_code):
        # Leave room group
        await self.channel_layer.group_discard(
            self.room_group_name,
            self.channel_name
        )

    # Receive message from WebSocket
    async def receive(self, text_data):
        try:
            text_data_json = json.loads(text_data)
            message_type = text_data_json.get('type')
            print("Received message_type: ", message_type)
            if message_type == 'get_pose_content':
                print("Received get_pose_content")
                await self.handle_get_file_content(text_data_json)

            else:
                message_content = text_data_json.get('message')

                if not message_type or not message_content:
                    raise ValueError("Invalid message format")

                await self.channel_layer.group_send(
                    self.room_group_name,
                    {
                        'type': 'send_message',
                        'message_type': message_type,
                        'message_content': message_content
                    }
                )
        except json.JSONDecodeError:
            logging.error("Received invalid JSON")
        except ValueError as e:
            logging.error(f"Received invalid message: {str(e)}")
        except Exception as e:
            logging.error(f"Error processing message: {str(e)}")

    # Receive message from room group
    async def send_message(self, event):
        # Send message to WebSocket
        await self.send(text_data=json.dumps({
            'type': event['message_type'],
            'message': event['message_content']
        }))

    async def send_poses(self, event):
        await self.send(text_data=json.dumps({
            'type': 'poses_meta',
            'poses': list(self.pose_meta.keys())
        }))

    async def handle_get_file_content(self, data):
        action_name = data.get('action_name')
        print("ACTION NAME: ", action_name)
        if not action_name:
            logging.error("Missing 'action_name' or 'file_name' in get_file_content message")
            return 

        file_path = self.pose_meta[action_name]
        if not os.path.exists(file_path):
            logging.error(f"File not found: {file_path}")
            await self.send(text_data=json.dumps({
                'type': 'pose_content',
                'error': 'File not found'
            }))
            return

        try:
            content = np.load(file_path)
            landmarks_list = [numpy_to_landmarks(frame) for frame in content]
            await self.send(text_data=json.dumps({
                'type': 'pose_content',
                'action_name': action_name,
                'frames': landmarks_list
            }))
        except Exception as e:
            logging.error(f"Error reading file: {str(e)}")
            await self.send(text_data=json.dumps({
                'type': 'pose_content',
                'error': 'Error reading file'
            }))

class WebcamConsumer(AsyncWebsocketConsumer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pose_data = []
        self.actions = []  # Store actions in memory (you might want to use a database in production)
        self.base_path = '../data/' 

    async def connect(self):
        self.holistic = mp.solutions.holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        await self.accept()
        await self.send_folder_info()  # Send folder info upon connection

    async def disconnect(self, close_code):
        # Leave room group
        print(f"WebSocket disconnected with code: {close_code}")

    async def receive(self, text_data):
        try:
            data = json.loads(text_data)
            message_type = data.get('type')

            if message_type == 'send_frame':
                await self.handle_image_message(data)
            elif message_type in ['create_action', 'update_action', 'delete_action']:
                await self.handle_action_message(data)
            elif message_type == 'save_frames':
                await self.handle_save_frames(data)
            elif message_type == 'get_file_content':
                await self.handle_get_file_content(data)
            elif message_type == 'delete_file':
                await self.handle_delete_file(data)
            else:
                logging.warning(f"Unknown message type: {message_type}")
        except json.JSONDecodeError:
            logging.error("Received invalid JSON")
        except Exception as e:
            logging.error(f"Error processing message: {str(e)}")

    async def handle_image_message(self, data):
        image_data = data['frame'].split(',')[1]
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        self.pose_data.append(img)
        if len(self.pose_data) == 30:
            # Process and send back the pose data
            await self.send_pose_data()
            self.pose_data = []  # Reset for next batch

    async def handle_action_message(self, data):
        message_type = data.get('type')
        action_name = data.get('name')

        if not message_type:
            logging.error("Missing 'type' in action message")
            return

        if message_type in ['create_action', 'update_action'] and not action_name:
            logging.error(f"Missing 'name' in {message_type} message")
            return

        if message_type == 'create_action':
            action_path = os.path.join(self.base_path, action_name)
            os.makedirs(action_path, exist_ok=True)

        elif message_type == 'update_action':
            old_name = data.get('old_name')
            if not old_name:
                logging.error("Missing 'old_name' in update_action message")
                return
            old_path = os.path.join(self.base_path, old_name)
            new_path = os.path.join(self.base_path, action_name)
            if os.path.exists(old_path):
                os.rename(old_path, new_path)

        elif message_type == 'delete_action':
            action_name = data.get('name')
            if not action_name:
                logging.error("Missing 'name' in delete_action message")
                return
            action_path = os.path.join(self.base_path, action_name)
            if os.path.exists(action_path):
                for file in os.listdir(action_path):
                    os.remove(os.path.join(action_path, file))
                os.rmdir(action_path)

        # Send updated folder info back to the client
        await self.send_folder_info()

    async def handle_save_frames(self, data):
        logging.info(f"Received save_frames message: {data.keys()}")
        action_name = data.get('action_name')
        frames = data.get('frames')

        if not action_name or not frames:
            logging.error("Missing 'action_name' or 'frames' in save_frames message")
            return

        action_path = os.path.join(self.base_path, action_name)
        if not os.path.exists(action_path):
            os.makedirs(action_path)

        keypoints_list = []
        for frame_data in frames:
            results = json.loads(frame_data)
            logging.info(f"Received results: {results.keys()}")
            keypoints = extract_keypoints_from_json(results)
            keypoints_list.append(keypoints)

        keypoints_array = np.array(keypoints_list)
        unique_id = uuid.uuid4().hex[:8]  # Generate a short unique identifier
        file_name = f"{action_name}_keypoints_{unique_id}.npy"
        file_path = os.path.join(action_path, file_name)
        np.save(file_path, keypoints_array)

        logging.info(f"Saved keypoints for {len(frames)} frames to {file_path}")
        await self.send_folder_info()

    async def handle_get_file_content(self, data):
        action_name = data.get('action_name')
        file_name = data.get('file_name')

        if not action_name or not file_name:
            logging.error("Missing 'action_name' or 'file_name' in get_file_content message")
            return

        file_path = os.path.join(self.base_path, action_name, file_name)
        if not os.path.exists(file_path):
            logging.error(f"File not found: {file_path}")
            await self.send(text_data=json.dumps({
                'type': 'file_content',
                'error': 'File not found'
            }))
            return

        try:
            content = np.load(file_path)
            landmarks_list = [numpy_to_landmarks(frame) for frame in content]
            await self.send(text_data=json.dumps({
                'type': 'file_content',
                'action_name': action_name,
                'file_name': file_name,
                'frames': landmarks_list
            }))
        except Exception as e:
            logging.error(f"Error reading file: {str(e)}")
            await self.send(text_data=json.dumps({
                'type': 'file_content',
                'error': 'Error reading file'
            }))

    async def handle_delete_file(self, data):
        action_name = data.get('action_name')
        file_name = data.get('file_name')

        if not action_name or not file_name:
            logging.error("Missing 'action_name' or 'file_name' in delete_file message")
            return

        file_path = os.path.join(self.base_path, action_name, file_name)
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                logging.info(f"Deleted file: {file_path}")
                await self.send_folder_info()
            except Exception as e:
                logging.error(f"Error deleting file: {str(e)}")
                await self.send(text_data=json.dumps({
                    'type': 'delete_file',
                    'error': 'Error deleting file'
                }))
        else:
            logging.error(f"File not found: {file_path}")
            await self.send(text_data=json.dumps({
                'type': 'delete_file',
                'error': 'File not found'
            }))

    async def send_pose_data(self):
        # Process pose data (you may want to add your pose estimation logic here)
        # For now, we'll just send back the raw frames
        results_list = []
        for frame in self.pose_data:
            _, results = mediapipe_detection(frame, self.holistic)
            results_list.append(results)
        await self.send(text_data=json.dumps({
            'type': 'pose_data',
            'frames': [output_to_json(results) for results in results_list]
        }))

    async def send_folder_info(self):
        folder_info = self.get_folder_info()
        await self.send(text_data=json.dumps({
            'type': 'folder_info',
            'data': folder_info
        }))

    def get_folder_info(self):
        folder_info = []
        for item in os.listdir(self.base_path):
            item_path = os.path.join(self.base_path, item)
            if os.path.isdir(item_path):
                files = [f for f in os.listdir(item_path) if os.path.isfile(os.path.join(item_path, f))]
                folder_info.append({
                    'name': item,
                    'files': files
                })
        return folder_info

    @classmethod
    async def decode_base64(cls, data):
        return base64.b64decode(data)
    
class SignDetectionConsumer(AsyncWebsocketConsumer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = logging.getLogger(__name__)

    async def connect(self):
        self.room_name = self.scope['url_route']['kwargs']['room_name']
        self.initiator = self.scope['url_route']['kwargs']['initiator']
        self.room_group_name = f'sign_detection_{self.room_name}_{self.initiator}'
        print(f"PRINT: Room name: {self.room_name}, Initiator: {self.initiator}, type: {type(self.initiator)}")
        # Join room group
        await self.channel_layer.group_add(
            self.room_group_name,
            self.channel_name
        )

        self.logger.info("SignDetectionConsumer: WebSocket connection established")
        self.holistic = mp.solutions.holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.model = load_model('../deeplearning/best_model.keras')
        self.label_to_text = {}
        with open('../deeplearning/label_to_text.json', 'r') as f:
            self.label_to_text = json.load(f)

        self.threshold = 0.75
        self.data_input = []
        self.predictions = None
        await self.accept()
        await self.send(text_data=json.dumps({
            'type': 'connection_status',
            'status': 'connected'
        }))

    async def disconnect(self, close_code):
        # Leave room group
        await self.channel_layer.group_discard(
            self.room_group_name,
            self.channel_name
        )
        self.logger.info(f"SignDetectionConsumer: WebSocket disconnected with code: {close_code}")

    async def receive(self, text_data):
        try:
            data = json.loads(text_data)
            message_type = data.get('type')

            if message_type == 'send_frame':
                await self.handle_image_message(data)
            else:
                logging.warning(f"Unknown message type: {message_type}")
        except json.JSONDecodeError:
            logging.error("Received invalid JSON")
        except Exception as e:
            logging.error(f"Error processing message: {str(e)}")
    
    async def handle_image_message(self, data):
        image_data = data['frame'].split(',')[1]
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        _, results = mediapipe_detection(img, self.holistic)
        keypoints = extract_keypoints_from_mediapipe(results)
        self.data_input.append(keypoints)

        if len(self.data_input) > 10:
            self.data_input = self.data_input[-10:]
        if len(self.data_input) == 10:
            input_data = np.expand_dims(self.data_input, axis=0)[:, :, -126:]
            input_data = preprocess_data(input_data)
            input_data = flip_data(input_data)
            res = self.model.predict(input_data, verbose = 0).mean(axis=0)

            if res[np.argmax(res)] >= self.threshold:
                word = self.label_to_text[str(np.argmax(res))]
            else:
                word = None
            
            if word is not None:
                if self.predictions is None or self.predictions != word:
                    self.predictions = word

        # Send to all members in the group
        await self.channel_layer.group_send(
            self.room_group_name,
            {
                'type': 'broadcast_pose_data',
                'frames': output_to_json(results),
                'predictions': self.predictions,
            }
        )

    async def broadcast_pose_data(self, event):
        # Send pose data to WebSocket
        await self.send(text_data=json.dumps({
            'type': 'pose_data',
            'frames': event['frames'],
            'predictions': event['predictions']
        }))

        # # await self.send(text_data=json.dumps({
        # #     'type': 'pose_data',
        # #     'frames': output_to_json(results)
        # # }))

        # await self.channel_layer.group_send(
        #     self.room_group_name,
        #     json.dumps({
        #         'type': 'pose_data',
        #         'frames': output_to_json(results)
        #     })
        # )

    @classmethod
    async def decode_base64(cls, data):
        return base64.b64decode(data)

        return base64.b64decode(data)
    