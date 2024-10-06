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
![](https://github.com/user-attachments/assets/fdb84fcb-56cf-4b8f-a9fd-254e04cf6049)

Turn on/off video/audio/pose tracking visualization:
![](https://github.com/user-attachments/assets/43a03089-0f62-435f-8a57-0018c0e26ad1)

2. Collect Data: 
Collect your own data + view your recorded pose before saving + manage data folders + view your collected data:
![](https://github.com/user-attachments/assets/bd85600b-28f1-48fd-a33f-27f9b7dd7e5d)

3. Train Model:
Train your own model + view training loss + view your model's performance on test data:
![](https://github.com/user-attachments/assets/32ea6386-8a16-40d2-a0d1-3374e45cb7b5)

4. Test Model:
Test your model + view the translation of hand gestures in real-time:
![](https://github.com/user-attachments/assets/4f50cb54-7778-4704-a157-dca5a81dbad6)
