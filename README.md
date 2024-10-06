# Video Call Web App with Realtime Sign Language Detection 🎥 🙋🏻‍♂️

This project is a video call web app that uses real-time sign language detection to translate hand gestures into text. 

It is built using:
- Frontend: React, TailwindCSS, WebRTC
- Backend: Django, Channels, and WebSockets
- Sign Language Detection: MediaPipe, TensorFlow

## Features
- Real-time sign language detection
- Text translation of hand gestures
- Video call and chat with two users
- Interface to collect and train your own dataset


## How to run
1. Clone the repository:
```
https://github.com/tranganhthuan/SignLanguageVideoCall.git
```
2. Install the dependencies
- Frontend:
```
cd frontend
npm install
```
- Backend:
```
# Python 3.12
cd backend
pip install -r requirements.txt
```

3. Run the development server
- Frontend:
```
cd frontend
npm start
```
- Backend:
```
cd backend
sh run.sh
```
- Training Model:
```
cd deeplearning
python train.py
```
- Test Model:
```
cd deeplearning
python inference.py
```

# Demo

1. Video Call:
Join the room + sign recognition + chat + view supported poses:
![Video Call](https://github.com/tranganhthuan/SignLanguageVideoCall/blob/main/assets/video_call.mp4)

Turn on/off video/audio/pose tracking visualization:
![Video Call 2](https://github.com/tranganhthuan/SignLanguageVideoCall/blob/main/assets/video_feature.mp4)

2. Collect Data: 
Collect your own data + view your recorded pose before saving + manage data folders + view your collected data:
![Collect Data](https://github.com/tranganhthuan/SignLanguageVideoCall/blob/main/assets/collect_data.mp4)

3. Train Model:
Train your own model + view training loss + view your model's performance on test data:
![Train Model](https://github.com/tranganhthuan/SignLanguageVideoCall/blob/main/assets/train_model.mp4)

4. Test Model:
Test your model + view the translation of hand gestures in real-time:
![Test Model](https://github.com/tranganhthuan/SignLanguageVideoCall/blob/main/assets/inference.mp4)





