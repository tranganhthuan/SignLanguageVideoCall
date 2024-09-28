import React, { useState, useEffect, useRef } from 'react';
import { useParams, Link } from 'react-router-dom';
import Peer from 'peerjs';
import useWebSocket from 'react-use-websocket';
import { drawConnectors, drawLandmarks } from "@mediapipe/drawing_utils";
import { FACEMESH_TESSELATION, FACEMESH_RIGHT_EYE, FACEMESH_RIGHT_EYEBROW, FACEMESH_LEFT_EYE, FACEMESH_LEFT_EYEBROW, FACEMESH_FACE_OVAL, FACEMESH_LIPS } from "@mediapipe/face_mesh";
import { HAND_CONNECTIONS } from "@mediapipe/hands";
import { POSE_CONNECTIONS } from "@mediapipe/pose";
import { MicrophoneIcon, VideoCameraIcon, SparklesIcon, PlayIcon, PauseIcon } from '@heroicons/react/24/solid';
import { MicrophoneIcon as MicrophoneIconOutline, VideoCameraIcon as VideoCameraIconOutline, SparklesIcon as SparklesIconOutline } from '@heroicons/react/24/outline';
import logo from '../logo.png';


const Room = () => {
  const { roomId, initiator } = useParams();
  const [roomName, setRoomName] = useState(roomId);
  const [localStream, setLocalStream] = useState(null);
  const [remoteStream, setRemoteStream] = useState(null);
  const [peer, setPeer] = useState(null);
  const [localId, setLocalId] = useState('');
  const [messages, setMessages] = useState([]);
  const [inputMessage, setInputMessage] = useState('');
  const [videoSize, setVideoSize] = useState({ width: 0, height: 0 });
  const [fileContentSizeAvailable, setFileContentSizeAvailable] = useState(false);

  const localVideoRef = useRef();
  const remoteVideoRef = useRef();
  const localCanvasRef = useRef(null);
  const remoteCanvasRef = useRef(null);
  const containerRef = useRef(null);
  const fileContentCanvasRef = useRef(null);
  const fileContentContainerRef = useRef(null);

  const socketRef = useRef();
  const isInitiator = (initiator === "true");
  const name = isInitiator ? "Host" : "Member";
  const remoteName = !isInitiator ? "Host" : "Member";
  const [isConnected, setIsConnected] = useState(false)
  const [posesMeta, setPosesMeta] = useState(null);
  const [selectedPose, setSelectedPose] = useState('');
  const [isPlayingPose, setIsPlayingPose] = useState(false);
  const [currentPoseFrame, setCurrentPoseFrame] = useState(0);
  const [selectedPoseContent, setSelectedPoseContent] = useState(null);
  const [localPredictions, setLocalPredictions] = useState('');
  const [remotePredictions, setRemotePredictions] = useState('');
  const [isLocalAudioOn, setIsLocalAudioOn] = useState(true);
  const [isLocalVideoOn, setIsLocalVideoOn] = useState(true);
  const [isLocalPoseDrawingOn, setIsLocalPoseDrawingOn] = useState(true);
  const [isRemoteAudioOn, setIsRemoteAudioOn] = useState(true);
  const [isRemoteVideoOn, setIsRemoteVideoOn] = useState(true);
  const [isRemotePoseDrawingOn, setIsRemotePoseDrawingOn] = useState(true);

  const messagesEndRef = useRef(null);

  const wss = useWebSocket(`ws://localhost:8000/ws/sign/${roomName}/${isInitiator}`, {
    onOpen: () => {
      console.log('WebSocket connection established');
      setIsConnected(true)
    },
    onError: (error) => {
      console.error('WebSocket error:', error);
    },
    onMessage: (event) => {
      try {
        const data = JSON.parse(event.data);
        // console.log("Parsed data:", data);
        if (data.type === 'pose_data') {
          drawResults(JSON.parse(data.frames), localCanvasRef.current, 4);
          setLocalPredictions(data.predictions);
        }
      } catch (error) {
        console.error('Error parsing message:', error);
      }
    },
    onClose: () => {
      console.log("WebSocket connection closed")
      setIsConnected(false)
    }
  });

  const not_wss = useWebSocket(`ws://localhost:8000/ws/sign/${roomName}/${!isInitiator}`, {
    onOpen: () => {
      console.log('Not WebSocket connection established');
    },
    onError: (error) => {
      console.error('Not WebSocket error:', error);
    },
    onMessage: (event) => {
      // console.log("NOTNOTNOT------")
      try {
        const data = JSON.parse(event.data);
        // console.log("NOTNOTNOTParsed data:", data);
        if (data.type === 'pose_data') {
          // console.log("(NOT WS)");
          // Ensure remoteVideoRef is ready before drawing
          if (remoteVideoRef.current && remoteVideoRef.current.videoWidth > 0) {
            console.log("Remote video now ready");
            drawResults(JSON.parse(data.frames), remoteCanvasRef.current, 4);
            setRemotePredictions(data.predictions);
          }
        }
        else if (data.type === 'shape') {
          console.log("(NOT WS) SHAPE", data.frame);
        }
      } catch (error) {
        console.error('Error parsing message:', error);
      }
    },
    onClose: () => {
      console.log("WebSocket connection closed")
    }
  });

  const drawResults = (results, canvas, lineWidth) => {
    if (!canvas) {
      console.error('Canvas is undefined');
      return;
    }

    const ctx = canvas.getContext('2d');
    if (!ctx) {
      console.error('Unable to get 2D context from canvas');
      return;
    }

    // Get the actual dimensions of the canvas
    const canvasWidth = canvas.clientWidth;
    const canvasHeight = canvas.clientHeight;

    // Set the canvas drawing dimensions to match its display size
    canvas.width = canvasWidth;
    canvas.height = canvasHeight;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Calculate scale factors based on the new dimensions
    const scaleX = canvasWidth / results.image_width;
    const scaleY = canvasHeight / results.image_height;

    // Apply scaling to the context
    ctx.scale(scaleX, scaleY);

    // Draw face landmarks
    if (results.face_landmarks) {
      drawConnectors(ctx, results.face_landmarks, FACEMESH_TESSELATION, { color: "#C0C0C070", lineWidth: 1 });
      drawConnectors(ctx, results.face_landmarks, FACEMESH_RIGHT_EYE, { color: "#30FF30" });
      drawConnectors(ctx, results.face_landmarks, FACEMESH_RIGHT_EYEBROW, { color: "#30FF30" });
      drawConnectors(ctx, results.face_landmarks, FACEMESH_LEFT_EYE, { color: "#30FF30" });
      drawConnectors(ctx, results.face_landmarks, FACEMESH_LEFT_EYEBROW, { color: "#30FF30" });
      drawConnectors(ctx, results.face_landmarks, FACEMESH_FACE_OVAL, { color: "#E0E0E0" });
      drawConnectors(ctx, results.face_landmarks, FACEMESH_LIPS, { color: "#E0E0E0" });
    }

    // Draw pose landmarks
    if (results.pose_landmarks) {
      drawConnectors(ctx, results.pose_landmarks, POSE_CONNECTIONS, { color: "#00FF00", lineWidth: lineWidth });
      drawLandmarks(ctx, results.pose_landmarks, { color: "#FF0000", lineWidth: lineWidth / 4 });
    }

    // Draw hand landmarks
    if (results.left_hand_landmarks) {
      drawConnectors(ctx, results.left_hand_landmarks, HAND_CONNECTIONS, { color: "#00FF00", lineWidth: lineWidth });
      drawLandmarks(ctx, results.left_hand_landmarks, { color: "#FF0000", lineWidth: lineWidth / 4 });
    }
    if (results.right_hand_landmarks) {
      drawConnectors(ctx, results.right_hand_landmarks, HAND_CONNECTIONS, { color: "#00FF00", lineWidth: lineWidth });
      drawLandmarks(ctx, results.right_hand_landmarks, { color: "#FF0000", lineWidth: lineWidth / 4 });
    }

    // Reset the scaling
    ctx.setTransform(1, 0, 0, 1, 0, 0);
  };

  const captureAndSendFrame = () => {
    const video = localVideoRef.current;
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');

    // Set canvas size to match the video's intrinsic size
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    // Draw the entire video frame onto the canvas at its original size
    ctx.drawImage(video, 0, 0, video.videoWidth, video.videoHeight);

    const frame = canvas.toDataURL('image/jpeg');
    wss.sendMessage(JSON.stringify({
      type: 'send_frame',
      frame: frame
    }));
  };

  useEffect(() => {
    let interval;
    if (isConnected) {
      interval = setInterval(captureAndSendFrame, 100);
    }
    return () => {
      if (interval) {
        clearInterval(interval);
      }
    };
  })

  useEffect(() => {
    const getMedia = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
        setLocalStream(stream);
        const NewPeer = createPeer(stream);
        setPeer(NewPeer);
      } catch (error) {
        console.error('Error accessing media devices:', error);
      }
    };

    getMedia();
  }, []);

  useEffect(() => {
    if (localId && localStream) {
      socketRef.current = new WebSocket(`ws://localhost:8000/ws/video/${roomName}/`);

      socketRef.current.onopen = () => {
        if (!isInitiator) {
          socketRef.current.send(JSON.stringify({ type: `${!isInitiator}`, message: localId }));
        }
      };

      socketRef.current.onmessage = (event) => {
        const data = JSON.parse(event.data);
        console.log("Received message:", data);

        if (data.type === initiator) {
          console.log("Attempting to initiate call", initiator, data.type, data.message);

          if (!peer) {
            console.error("Peer object is not initialized");
            return;
          }

          try {
            var call = peer.call(data.message, localStream);

            if (!call) {
              console.error("Failed to create call object");
              return;
            }

            call.on('stream', (remoteStream) => {
              setRemoteStream(remoteStream);
              if (remoteVideoRef.current) {
                remoteVideoRef.current.srcObject = remoteStream;
              } else {
                console.warn("Remote video ref is not available");
              }
            });

            call.on('error', (err) => {
              console.error("Call error:", err);
            });

          } catch (error) {
            console.error("Error initiating call:", error);
          }

          if (isInitiator) {
            socketRef.current.send(JSON.stringify({ type: `${!isInitiator}`, message: localId }));
          }
        } else if (data.type === "chat") {
          setMessages((prevMessages) => [...prevMessages, data.message]);
        } else if (data.type === "poses_meta") {
          console.log("Received poses meta:", data.poses);
          setPosesMeta(data.poses);
        } else if (data.type === "pose_content") {
          console.log("Received pose content:", data.frames);
          setSelectedPoseContent(data.frames);
          // if (fileContentCanvasRef.current) {
          //   drawResults(data.frames[0], fileContentCanvasRef.current);
          // }
        }

      };

      socketRef.current.onerror = (error) => {
        console.error("WebSocket error:", error);
      };

      socketRef.current.onclose = (event) => {
        console.warn("WebSocket closed:", event);
      };
    }
  }, [localId, localStream])

  useEffect(() => {
    return () => {
      if (localStream) {
        localStream.getTracks().forEach(track => track.stop());
      }
      if (socketRef.current) {
        socketRef.current.close();
      }
      if (peer) {
        peer.destroy();
      }
    };
  }, []);

  useEffect(() => {
    if (localVideoRef.current && localStream) {
      localVideoRef.current.srcObject = localStream;
    }
  }, [localStream]);

  useEffect(() => {
    if (remoteVideoRef.current && remoteStream) {
      remoteVideoRef.current.srcObject = remoteStream;
    }
  }, [remoteStream]);

  useEffect(() => {
    const updateVideoSize = () => {
      if (containerRef.current) {
        const containerWidth = containerRef.current.offsetWidth;
        const containerHeight = containerRef.current.offsetHeight;
        const videoWidth = containerWidth / 2; // 50% of container width
        const videoHeight = containerHeight;

        setVideoSize({ width: videoWidth, height: videoHeight });

        // Update canvas sizes
        if (localCanvasRef.current) {
          localCanvasRef.current.width = videoWidth;
          localCanvasRef.current.height = videoHeight;
        }

        if (remoteCanvasRef.current) {
          remoteCanvasRef.current.width = videoWidth;
          remoteCanvasRef.current.height = videoHeight;
        }
      }
    };

    updateVideoSize();
    window.addEventListener('resize', updateVideoSize);

    return () => {
      window.removeEventListener('resize', updateVideoSize);
    };
  }, []);

  const createPeer = (stream) => {
    const peer = new Peer();
    console.log("Create PEER", initiator);
    peer.on('open', (id) => {
      setLocalId(id);
    });

    peer.on('call', (call) => {
      console.log(`Received call: ${isInitiator}`);
      call.answer(stream);
      call.on('stream', (remoteStream) => {
        setRemoteStream(remoteStream);
        if (remoteVideoRef.current) {
          remoteVideoRef.current.srcObject = remoteStream;
        }
      });
    });

    return peer;
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    console.log("send message chat");
    if (inputMessage.trim() !== '' && socketRef.current.readyState === WebSocket.OPEN) {
      socketRef.current.send(JSON.stringify({
        'type': 'chat',
        'message': `${name}: ${inputMessage}`
      }));
      setInputMessage('');
    }
  };

  const handlePoseSelect = (e) => {
    const selectedPoseName = e.target.value;
    setSelectedPose(selectedPoseName);
    setIsPlayingPose(false);
    setCurrentPoseFrame(0);

    // Fetch the pose content from the server
    if (selectedPoseName) {
      console.log(selectedPose)
      socketRef.current.send(JSON.stringify({
        'type': 'get_pose_content',
        'action_name': selectedPoseName
      }));
      console.log("Sent get_pose_content")
    } else {
      setSelectedPoseContent(null);
    }
  };

  useEffect(() => {
    if (fileContentCanvasRef.current && fileContentContainerRef.current) {
      const canvas = fileContentCanvasRef.current;
      const container = fileContentContainerRef.current;

      // Get the client dimensions of the container
      const containerWidth = container.clientWidth;
      const containerHeight = container.clientHeight;

      console.log("containerWidth:", containerWidth);
      console.log("containerHeight:", containerHeight);

      // Calculate aspect ratio to maintain
      const aspectRatio = 4 / 3;

      // Calculate new dimensions while maintaining aspect ratio
      let newWidth, newHeight;
      if (containerWidth / containerHeight > aspectRatio) {
        newHeight = containerHeight;
        newWidth = newHeight * aspectRatio;
      } else {
        newWidth = containerWidth;
        newHeight = newWidth / aspectRatio;
      }

      // Set canvas dimensions
      canvas.width = newWidth;
      canvas.height = newHeight;

      console.log("aaa:", newWidth);
      console.log("bbb:", newHeight);
      console.log("New width:", canvas.width);
      console.log("New height:", canvas.height);

      // Set canvas style dimensions to match
      canvas.style.width = `${newWidth}px`;
      canvas.style.height = `${newHeight}px`;

      try {
        const frameData = typeof selectedPoseContent[0] === 'object' ? selectedPoseContent[0] : JSON.parse(selectedPoseContent[0]);
        console.log("Canvas dimensions:", canvas.width, canvas.height);
        console.log("Canvas client dimensions:", canvas.clientWidth, canvas.clientHeight);
        setFileContentSizeAvailable(true);
        drawResults(frameData, canvas, 1);
      } catch (error) {
        console.error('Error parsing pose content:', error);
      }
    }
  }, [fileContentContainerRef, fileContentContainerRef, selectedPoseContent]);
  useEffect(() => {
    let timeoutId;
    if (isPlayingPose && selectedPoseContent) {
      const playFrame = () => {
        if (currentPoseFrame < selectedPoseContent.length - 1) {
          setCurrentPoseFrame(prev => prev + 1);
          timeoutId = setTimeout(playFrame, 100); // 100ms for 10 FPS
        } else {
          setIsPlayingPose(false);
          setCurrentPoseFrame(0);
        }
      };
      timeoutId = setTimeout(playFrame, 100);
    }
    return () => clearTimeout(timeoutId);
  }, [isPlayingPose, currentPoseFrame, selectedPoseContent]);

  useEffect(() => {
    if (selectedPoseContent && selectedPoseContent[currentPoseFrame] && fileContentCanvasRef.current && fileContentSizeAvailable) {
      drawResults(selectedPoseContent[currentPoseFrame], fileContentCanvasRef.current, 1);
    }
  }, [currentPoseFrame, selectedPoseContent]);

  const handlePosePlay = () => setIsPlayingPose(true);
  const handlePosePause = () => setIsPlayingPose(false);

  const handlePoseSeek = (e) => {
    const seekPosition = parseInt(e.target.value);
    setCurrentPoseFrame(seekPosition);
    if (selectedPoseContent && selectedPoseContent[seekPosition] && fileContentCanvasRef.current && fileContentSizeAvailable) {
      drawResults(selectedPoseContent[seekPosition], fileContentCanvasRef.current, 1);
    }
  };

  const toggleLocalAudio = () => {
    if (localStream) {
      localStream.getAudioTracks().forEach(track => track.enabled = !isLocalAudioOn);
      setIsLocalAudioOn(!isLocalAudioOn);
    }
  };

  const toggleLocalVideo = () => {
    if (localStream) {
      localStream.getVideoTracks().forEach(track => track.enabled = !isLocalVideoOn);
      setIsLocalVideoOn(!isLocalVideoOn);
    }
  };

  const toggleRemoteAudio = () => {
    if (remoteVideoRef.current) {
      remoteVideoRef.current.muted = isRemoteAudioOn;
      setIsRemoteAudioOn(!isRemoteAudioOn);
    }
  };

  const toggleRemoteVideo = () => {
    setIsRemoteVideoOn(!isRemoteVideoOn);
  };

  const toggleLocalPoseDrawing = () => {
    setIsLocalPoseDrawingOn(!isLocalPoseDrawingOn);
  };

  const toggleRemotePoseDrawing = () => {
    setIsRemotePoseDrawingOn(!isRemotePoseDrawingOn);
  };

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }

  useEffect(scrollToBottom, [messages]);

  return (
    <div className="flex flex-col h-screen bg-gray-100">
      {/* Navbar */}
      <nav className="bg-white shadow-md">
        <div className="container flex items-center">
          <Link to="/" className="flex items-center">
            <img src={logo} alt="Logo" className="ml-4 mr-2 my-2 h-8" />
            <h1 className="text-l font-bold text-gray-800">Sign Language Video Call App</h1>
          </Link>
        </div>
      </nav>

      {/* Existing content */}
      <div ref={containerRef} className="flex-grow flex h-full">
        <div className="w-1/2 p-2">
          <div className="relative w-full h-full border border-gray-300 rounded-lg">
            <video
              ref={localVideoRef}
              className="absolute top-0 left-0 w-full h-full rounded-lg object-cover"
              autoPlay
              muted
            />
            <canvas
              ref={localCanvasRef}
              className="absolute top-0 left-0 w-full h-full"
              style={{ display: isLocalPoseDrawingOn ? 'block' : 'none' }}
            />
            <div className="absolute top-2 left-2 right-2 bg-black bg-opacity-50 p-2 rounded">
              <h1 className="text-m text-white">
                Role: {name} - Predicted text: {localPredictions}
              </h1>
            </div>
            <div className="absolute bottom-4 left-1/2 transform -translate-x-1/2 flex space-x-4">
              <button
                onClick={toggleLocalAudio}
                className="bg-gray-800 p-3 rounded-full hover:bg-gray-700 focus:outline-none"
              >
                {isLocalAudioOn ?
                  <MicrophoneIcon className="h-6 w-6 text-white" /> :
                  <MicrophoneIconOutline className="h-6 w-6 text-gray-400" />
                }
              </button>
              <button
                onClick={toggleLocalVideo}
                className="bg-gray-800 p-3 rounded-full hover:bg-gray-700 focus:outline-none"
              >
                {isLocalVideoOn ?
                  <VideoCameraIcon className="h-6 w-6 text-white" /> :
                  <VideoCameraIconOutline className="h-6 w-6 text-gray-400" />
                }
              </button>
              <button
                onClick={toggleLocalPoseDrawing}
                className="bg-gray-800 p-3 rounded-full hover:bg-gray-700 focus:outline-none"
              >
                {isLocalPoseDrawingOn ?
                  <SparklesIcon className="h-6 w-6 text-white" /> :
                  <SparklesIconOutline className="h-6 w-6 text-gray-400" />
                }
              </button>
            </div>
          </div>
        </div>
        <div className="w-1/2 p-2">
          <div className="relative w-full h-full border border-gray-300 rounded-lg">
            <video
              ref={remoteVideoRef}
              className="absolute top-0 left-0 w-full h-full rounded-lg object-cover"
              autoPlay
              style={{ display: isRemoteVideoOn ? 'block' : 'none' }}
            />
            <canvas
              ref={remoteCanvasRef}
              className="absolute top-0 left-0 w-full h-full"
              style={{ display: isRemotePoseDrawingOn && isRemoteVideoOn ? 'block' : 'none' }}
            />
            <div className="absolute top-2 left-2 right-2 bg-black bg-opacity-50 p-2 rounded">
              <h1 className="text-m text-white">
                Role: {remoteName} - Predicted text: {remotePredictions}
              </h1>
            </div>
            <div className="absolute bottom-4 left-1/2 transform -translate-x-1/2 flex space-x-4">
              <button
                onClick={toggleRemoteAudio}
                className="bg-gray-800 p-3 rounded-full hover:bg-gray-700 focus:outline-none"
              >
                {isRemoteAudioOn ?
                  <MicrophoneIcon className="h-6 w-6 text-white" /> :
                  <MicrophoneIconOutline className="h-6 w-6 text-gray-400" />
                }
              </button>
              <button
                onClick={toggleRemoteVideo}
                className="bg-gray-800 p-3 rounded-full hover:bg-gray-700 focus:outline-none"
              >
                {isRemoteVideoOn ?
                  <VideoCameraIcon className="h-6 w-6 text-white" /> :
                  <VideoCameraIconOutline className="h-6 w-6 text-gray-400" />
                }
              </button>
              <button
                onClick={toggleRemotePoseDrawing}
                className="bg-gray-800 p-3 rounded-full hover:bg-gray-700 focus:outline-none"
              >
                {isRemotePoseDrawingOn ?
                  <SparklesIcon className="h-6 w-6 text-white" /> :
                  <SparklesIconOutline className="h-6 w-6 text-gray-400" />
                }
              </button>
            </div>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-2 h-96">
        <div className="h-full p-2">
          <div className="flex flex-col h-full bg-white rounded-lg shadow p-2">
            <p className="text-m ml-2 mb-2">Chat</p>
            <hr className="h-px bg-gray-200 border-0 dark:bg-gray-700" />
            <div className="flex-grow overflow-y-auto p-2 mb-2 h-50">
              <div className="h-full overflow-y-auto">
                {messages.map((message, index) => (
                  <p key={index} className="text-sm bg-gray-200 p-2 rounded-lg mb-1">
                    {message}
                  </p>
                ))}
                <div ref={messagesEndRef} />
              </div>
            </div>
            <form onSubmit={handleSubmit} className="flex">
              <input
                type="text"
                value={inputMessage}
                onChange={(e) => setInputMessage(e.target.value)}
                className="flex-grow p-2 border border-gray-300 rounded-l-lg focus:outline-none"
              />
              <button
                type="submit"
                className="bg-gray-800 hover:bg-gray-700 text-white px-4 rounded-r-lg hover:bg-blue-600"
              >
                Send
              </button>
            </form>
          </div>
        </div>

        <div className='h-full p-2'>
          <div className="flex flex-col h-full bg-white rounded-lg shadow p-2">
            {posesMeta && (
              <div className="mt-2 flex h-full">
                <div className="w-1/3 pr-4">
                  <p className="text-m ml-2 mb-2">Sign Language</p>
                  <hr className="h-px mb-2 bg-gray-200 border-0 dark:bg-gray-700" />
                  <select
                    value={selectedPose}
                    onChange={handlePoseSelect}
                    className="w-full p-2 border border-gray-300 rounded-lg focus:outline-none mb-4"
                  >
                    <option value="">Select a pose</option>
                    {posesMeta.map((pose, index) => (
                      <option key={index} value={pose}>
                        {pose}
                      </option>
                    ))}
                  </select>
                  {selectedPose && (
                    <div className="flex flex-col space-y-2">
                      <div className="flex items-center space-x-2">
                        <button
                          className="bg-gray-800 text-white p-3 rounded-full hover:bg-gray-700"
                          onClick={isPlayingPose ? handlePosePause : handlePosePlay}
                          disabled={!selectedPoseContent}
                        >
                          {isPlayingPose ?
                            <PauseIcon className="h-6 w-6" /> :
                            <PlayIcon className="h-6 w-6" />
                          }
                        </button>
                        <input
                          type="range"
                          min="0"
                          max={selectedPoseContent ? selectedPoseContent.length - 1 : 0}
                          value={currentPoseFrame}
                          onChange={handlePoseSeek}
                          className="flex-grow accent-gray-800 hover:accent-gray-700"
                        />
                      </div>
                    </div>
                  )}
                </div>
                <div className="w-2/3 h-full flex-grow" ref={fileContentContainerRef}>
                  <div className="relative w-full h-full">
                    <canvas className="mx-auto"
                      ref={fileContentCanvasRef}
                    />
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default Room;