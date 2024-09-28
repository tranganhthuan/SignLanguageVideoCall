import React, { useEffect, useRef, useState } from 'react';
import { useParams, Link } from 'react-router-dom';
import useWebSocket from 'react-use-websocket';
import { ChevronDownIcon, ChevronRightIcon, PencilIcon, TrashIcon, PlayIcon, PauseIcon, FolderIcon, VideoCameraIcon } from '@heroicons/react/24/solid';
import { drawConnectors, drawLandmarks } from "@mediapipe/drawing_utils";
import { FACEMESH_TESSELATION, FACEMESH_RIGHT_EYE, FACEMESH_RIGHT_EYEBROW, FACEMESH_LEFT_EYE, FACEMESH_LEFT_EYEBROW, FACEMESH_FACE_OVAL, FACEMESH_LIPS } from "@mediapipe/face_mesh";
import { HAND_CONNECTIONS } from "@mediapipe/hands";
import { POSE_CONNECTIONS } from "@mediapipe/pose";

import logo from '../logo.png';

const Video = () => {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [isRecording, setIsRecording] = useState(false);
  const [frameCount, setFrameCount] = useState(0);
  const maxFrames = 30; // Set this to your desired {n} value
  const [poseFrames, setPoseFrames] = useState([]);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentFrame, setCurrentFrame] = useState(0);
  const fps = 10; // Fixed FPS at 10
  const intervalRef = useRef(null);

  const [newAction, setNewAction] = useState('');
  const [editingIndex, setEditingIndex] = useState(null);
  const [expandedIndex, setExpandedIndex] = useState(null);
  const [folderInfo, setFolderInfo] = useState([]);

  const [selectedFile, setSelectedFile] = useState(null);
  const [fileContent, setFileContent] = useState(null);
  const fileContentCanvasRef = useRef(null);

  const [isPlayingFile, setIsPlayingFile] = useState(false);
  const [currentFileFrame, setCurrentFileFrame] = useState(0);
  const fileFps = 10; // Fixed file FPS at 10
  const fileIntervalRef = useRef(null);

  const [dimensions, setDimensions] = useState({ width: 0, height: 0 });
  const [fileDimensions, setFileDimensions] = useState({ width: 0, height: 0 });
  const containerRef = useRef(null);
  const mainContainerRef = useRef(null);
  const navContainerRef = useRef(null);
  const upperContainerRef = useRef(null);
  const fileContentRef = useRef(null);
  const fileContentSubContainerRef = useRef(null);
  const fileContentContainerRef = useRef(null);
  const fileContentTitleRef = useRef(null);
  const [remainingHeight, setRemainingHeight] = useState(0);

  useEffect(() => {
    const updateDimensions = () => {
      if (containerRef.current) {
        const width = containerRef.current.offsetWidth / 2; // Half of the container width
        const height = width * (9 / 16); // 16:9 aspect ratio
        setDimensions({ width, height });
      }
    };

    updateDimensions();
    window.addEventListener('resize', updateDimensions);
    return () => window.removeEventListener('resize', updateDimensions);
  }, []);

  useEffect(() => {
    console.log("fileContentCanvasRef", fileContentCanvasRef.current)
    console.log("fileContentCanvasRef.clientHeight", fileContentCanvasRef.current.clientHeight)
    console.log("fileContentContainerRef.clientHeight", fileContentContainerRef.current.clientHeight)
    if (fileContentContainerRef.current.clientHeight > 0) {
      const height = fileContentSubContainerRef.current.clientHeight
      const width = height * (16 / 9)
      setFileDimensions({ width, height })
      console.log("width", width)
      console.log("height", height)
      fileContentCanvasRef.current.width = width
      fileContentCanvasRef.current.height = height
      fileContentCanvasRef.current.style.width = `${width}px`
      fileContentCanvasRef.current.style.height = `${height}px`
    }
  }, [fileContentCanvasRef, remainingHeight]);

  useEffect(() => {
    const updateRemainingHeight = () => {
      if (mainContainerRef.current && containerRef.current) {
        const totalHeight = mainContainerRef.current.offsetHeight;
        const navContainerHeight = navContainerRef.current.offsetHeight;
        const upperContentHeight = upperContainerRef.current.offsetHeight;
        const newRemainingHeight = totalHeight - navContainerHeight - upperContentHeight;
        setRemainingHeight(newRemainingHeight);
      }
    };

    updateRemainingHeight();
    window.addEventListener('resize', updateRemainingHeight);
    return () => window.removeEventListener('resize', updateRemainingHeight);
  }, []);

  const ws = useWebSocket('ws://127.0.0.1:8000/ws/webcam/', {
    onOpen: () => {
      console.log('WebSocket connection established');
    },
    onError: (error) => {
      console.error('WebSocket error:', error);
    },
    onMessage: (event) => {
      try {
        const data = JSON.parse(event.data);
        console.log("Parsed data:", data);
        if (data.type === 'pose_data') {
          setPoseFrames(data.frames);
          if (data.frames.length > 0 && canvasRef.current) {
            drawResults(JSON.parse(data.frames[0]), canvasRef.current);
          }
        } else if (data.type === 'folder_info') {
          console.log("Folder info:", data.data);
          setFolderInfo(data.data);
        } else if (data.type === 'file_content') {
          if (data.error) {
            console.error('Error getting file content:', data.error);
          } else {
            setFileContent(data.frames);
            console.log('Received file content:', data.frames[0]);
            if (fileContentCanvasRef.current) {
              drawResults(data.frames[0], fileContentCanvasRef.current);
            }
          }
        }
      } catch (error) {
        console.error('Error parsing message:', error);
      }
    },
  });

  useEffect(() => {
    navigator.mediaDevices.getUserMedia({ video: true })
      .then(stream => {
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
      })
      .catch(err => console.error("Error accessing webcam:", err));
  }, []);

  useEffect(() => {
    const interval = setInterval(captureAndSendFrame, 100); // Adjust interval as needed
    return () => clearInterval(interval);
  }, [isRecording, frameCount]);

  const captureAndSendFrame = () => {
    if (!isRecording || frameCount >= maxFrames) return;

    const canvas = document.createElement('canvas');
    canvas.width = videoRef.current.videoWidth;
    canvas.height = videoRef.current.videoHeight;
    canvas.getContext('2d').drawImage(videoRef.current, 0, 0);
    const frame = canvas.toDataURL('image/jpeg');
    console.log("sending frame")
    ws.sendMessage(JSON.stringify({
      type: 'send_frame',
      frame: frame
    }));

    setFrameCount(prevCount => {
      if (prevCount + 1 >= maxFrames) {
        setIsRecording(false);
      }
      return prevCount + 1;
    });
  };

  const playFrames = () => {
    if (currentFrame < poseFrames.length) {
      if (poseFrames[currentFrame] && canvasRef.current) {
        console.log("Drawing frame", currentFrame);
        drawResults(JSON.parse(poseFrames[currentFrame]), canvasRef.current);
        setCurrentFrame(prev => prev + 1);
      } else {
        console.error("Invalid frame data at index", currentFrame);
        setCurrentFrame(prev => prev + 1);
      }
    } else {
      setIsPlaying(false);
      setCurrentFrame(0);
      console.log("Finished playing all frames");
    }
  };

  useEffect(() => {
    if (isPlaying) {
      intervalRef.current = setInterval(playFrames, 1000 / fps);
    } else {
      clearInterval(intervalRef.current);
    }
    return () => clearInterval(intervalRef.current);
  }, [isPlaying, currentFrame]);

  const handlePlay = () => setIsPlaying(true);
  const handlePause = () => setIsPlaying(false);

  const handleSeek = (e) => {
    const seekPosition = parseInt(e.target.value);
    setCurrentFrame(seekPosition);
    if (poseFrames[seekPosition] && canvasRef.current) {
      drawResults(JSON.parse(poseFrames[seekPosition]), canvasRef.current);
    }
  };

  const handleRecordClick = () => {
    setIsRecording(true);
    setFrameCount(0);
    setCurrentFrame(0);
  };

  const handleSave = () => {
    if (expandedIndex === null || expandedIndex >= folderInfo.length) {
      console.error("No valid action folder selected");
      // You might want to show an alert or notification to the user here
      return;
    }

    const selectedAction = folderInfo[expandedIndex].name;
    console.log("Saving frames to action:", selectedAction);

    // poseFrames already contain the correct pose data, no need to format
    ws.sendMessage(JSON.stringify({
      type: 'save_frames',
      action_name: selectedAction,
      frames: poseFrames
    }));
  };

  const handleAddAction = () => {
    if (newAction.trim()) {
      ws.sendMessage(JSON.stringify({
        type: 'create_action',
        name: newAction.trim()
      }));
      setNewAction('');
    }
  };

  const handleEditAction = (index) => {
    setEditingIndex(index);
    setNewAction(folderInfo[index].name);
  };

  const handleUpdateAction = () => {
    if (newAction.trim() && editingIndex !== null) {
      ws.sendMessage(JSON.stringify({
        type: 'update_action',
        index: editingIndex,
        old_name: folderInfo[editingIndex].name,
        name: newAction.trim()
      }));
      setNewAction('');
      setEditingIndex(null);
    }
  };

  const handleDeleteAction = (index) => {
    console.log("Deleting action at index", index);
    ws.sendMessage(JSON.stringify({
      type: 'delete_action',
      index: index,
      name: folderInfo[index].name
    }));
  };

  const toggleExpand = (index) => {
    setExpandedIndex(expandedIndex === index ? null : index);
  };

  const handleFileSelect = (actionName, fileName) => {
    setSelectedFile({ actionName, fileName });
    getFileContent(actionName, fileName);
  };

  const getFileContent = (actionName, fileName) => {
    ws.sendMessage(JSON.stringify({
      type: 'get_file_content',
      action_name: actionName,
      file_name: fileName
    }));
  };

  useEffect(() => {
    if (fileContent && fileContent.length > 0 && fileContentCanvasRef.current) {
      const canvas = fileContentCanvasRef.current;
      canvas.width = dimensions.width;
      canvas.height = dimensions.height;

      try {
        const frameData = typeof fileContent[0] === 'object' ? fileContent[0] : JSON.parse(fileContent[0]);
        drawResults(frameData, canvas);
      } catch (error) {
        console.error('Error parsing file content:', error);
      }
    }
  }, [fileContent, dimensions]);

  const handleFilePlay = () => setIsPlayingFile(true);
  const handleFilePause = () => setIsPlayingFile(false);

  const handleFileSeek = (e) => {
    const seekPosition = parseInt(e.target.value);
    setCurrentFileFrame(seekPosition);
    console.log("Current file frame:", fileContent[seekPosition]);
    if (fileContent && fileContent[seekPosition] && fileContentCanvasRef.current) {
      drawResults(fileContent[seekPosition], fileContentCanvasRef.current);
    }
  };

  useEffect(() => {
    if (isPlayingFile && fileContent && fileContentCanvasRef.current) {
      fileIntervalRef.current = setInterval(() => {
        setCurrentFileFrame(prev => {
          const next = prev + 1;
          if (next < fileContent.length) {
            drawResults(fileContent[next], fileContentCanvasRef.current);
            return next;
          } else {
            setIsPlayingFile(false);
            return 0; // Reset to 0 when reaching the end
          }
        });
      }, 1000 / fileFps);
    } else {
      clearInterval(fileIntervalRef.current);
    }
    return () => clearInterval(fileIntervalRef.current);
  }, [isPlayingFile, fileContent]);

  const handleDeleteFile = (actionName, fileName) => {
    console.log(`Deleting file: ${fileName} from action: ${actionName}`);
    ws.sendMessage(JSON.stringify({
      type: 'delete_file',
      action_name: actionName,
      file_name: fileName
    }));
  };

  const drawResults = (results, canvas) => {
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Update canvas dimensions
    canvas.width = canvas === canvasRef.current ? dimensions.width : fileDimensions.width;
    canvas.height = canvas === canvasRef.current ? dimensions.height : fileDimensions.height;
    console.log('fileContentCanvasRef', fileDimensions.height, fileDimensions.width);
    console.log("Canvas dimensions:", canvas.width, canvas.height);
    ctx.clearRect(0, 0, canvas.width, canvas.height);

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
      drawConnectors(ctx, results.pose_landmarks, POSE_CONNECTIONS, { color: "#00FF00", lineWidth: 4 });
      drawLandmarks(ctx, results.pose_landmarks, { color: "#FF0000", lineWidth: 2 });
    }

    // Draw hand landmarks
    if (results.left_hand_landmarks) {
      drawConnectors(ctx, results.left_hand_landmarks, HAND_CONNECTIONS, { color: "#00FF00", lineWidth: 5 });
      drawLandmarks(ctx, results.left_hand_landmarks, { color: "#FF0000", lineWidth: 2 });
    }
    if (results.right_hand_landmarks) {
      drawConnectors(ctx, results.right_hand_landmarks, HAND_CONNECTIONS, { color: "#00FF00", lineWidth: 5 });
      drawLandmarks(ctx, results.right_hand_landmarks, { color: "#FF0000", lineWidth: 2 });
    }
  };

  return (
    <div className="flex flex-col h-screen bg-gray-100" ref={mainContainerRef}>
      {/* Navbar */}
      <nav className="bg-white shadow-md" ref={navContainerRef}>
        <div className="container flex items-center">
          <Link to="/" className="flex items-center">
            <img src={logo} alt="Logo" className="ml-4 mr-2 my-2 h-8" />
            <h1 className="text-l font-bold text-gray-800">Sign Language Video Call App</h1>
          </Link>
        </div>
      </nav>

      {/* Main content */}
      <div className="flex flex-wrap flex-grow" ref={containerRef}>
        {/* Video and Pose wrapper */}
        <div className="w-full flex flex-wrap h-fit" ref={upperContainerRef}>
          {/* Video */}
          <div className="w-1/2 p-2">
            <div className="relative border border-gray-300 rounded-lg">
              <video
                ref={videoRef}
                className="w-full object-cover rounded-lg"
                autoPlay
                muted
                style={{ aspectRatio: '16/9' }}
              />
              <p className="absolute top-2 left-2 text-m text-white bg-black bg-opacity-50 px-2 py-1 rounded">Video</p>
              <div className="absolute bottom-2 right-2 flex space-x-2">
                <button
                  className={`bg-gray-800 p-3 rounded-full hover:bg-gray-700 focus:outline-none ${isRecording ? 'bg-red-500 hover:bg-red-500' : 'bg-gray-800 hover:bg-gray-700'}`}
                  onClick={handleRecordClick}
                  disabled={isRecording}
                >
                  <VideoCameraIcon className="h-6 w-6 text-white" />
                </button>
              </div>
            </div>
          </div>

          {/* Pose */}
          <div className="w-1/2 p-2">
            <div className="relative border border-gray-300 rounded-lg">
              <canvas
                ref={canvasRef}
                className="w-full object-cover rounded-lg"
                style={{ aspectRatio: '16/9' }}
              />
              <p className="absolute top-2 left-2 text-m text-white bg-black bg-opacity-50 px-2 py-1 rounded">Pose</p>
              <div className="mt-2 flex flex-col space-y-2 absolute bottom-2 right-2 left-2">
                <div className="flex items-center space-x-2">
                  <button
                    className="bg-gray-800 p-3 rounded-full hover:bg-gray-700 focus:outline-none"
                    onClick={isPlaying ? handlePause : handlePlay}
                    disabled={poseFrames.length === 0}
                  >
                    {isPlaying ? (
                      <PauseIcon className="h-6 w-6 text-white" />
                    ) : (
                      <PlayIcon className="h-6 w-6 text-white" />
                    )}
                  </button>
                  <button
                    className="bg-gray-800 p-3 rounded-full hover:bg-gray-700 focus:outline-none disabled:bg-gray-400"
                    onClick={handleSave}
                    disabled={poseFrames.length === 0 || expandedIndex === null || expandedIndex >= folderInfo.length}
                  >
                    <FolderIcon className="h-6 w-6 text-white" />
                  </button>
                  <input
                    type="range"
                    min="0"
                    max={poseFrames.length - 1}
                    value={currentFrame}
                    onChange={handleSeek}
                    className="flex-grow accent-gray-800 hover:accent-gray-700"
                  />
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Folder Management and File Content container */}
        <div className="w-full flex flex-wrap" style={{ height: `${remainingHeight}px` }}>
          {/* Folder Management */}
          <div className="w-1/2 p-2 h-full">
            <div className="h-full bg-white border border-gray-300 rounded-lg p-2 flex flex-col">
              <p className="text-m mb-2">Folder Management</p>
              <div className="flex mb-2">
                <input
                  type="text"
                  value={newAction}
                  onChange={(e) => setNewAction(e.target.value)}
                  className="flex-grow px-4 py-2 border rounded-l-md"
                  placeholder="Enter new action"
                />
                <button
                  onClick={editingIndex !== null ? handleUpdateAction : handleAddAction}
                  className="bg-gray-800 p-3 hover:bg-gray-700 text-white px-4 py-2 rounded-r-md"
                >
                  {editingIndex !== null ? 'Update' : 'Add'} Action
                </button>
              </div>
              <div className="overflow-y-auto flex-grow bg-white p-2 rounded-md shadow" style={{ height: 'calc(100% - 6rem)' }}>
                <ul className="space-y-2">
                  {folderInfo.map((folder, index) => (
                    <li key={index} className="bg-gray-100 p-2 rounded">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center">
                          <button onClick={() => toggleExpand(index)} className="mr-2">
                            {expandedIndex === index ? (
                              <ChevronDownIcon className="h-5 w-5" />
                            ) : (
                              <ChevronRightIcon className="h-5 w-5" />
                            )}
                          </button>
                          <span>{folder.name} - {folder.files.length}</span>
                        </div>
                        <div>
                          <button onClick={() => handleEditAction(index)} className="text-gray-800 hover:text-gray-700 mr-2">
                            <PencilIcon className="h-5 w-5" />
                          </button>
                          <button onClick={() => handleDeleteAction(index)} className="text-gray-800 hover:text-gray-700">
                            <TrashIcon className="h-5 w-5" />
                          </button>
                        </div>
                      </div>
                      {expandedIndex === index && (
                        <div className="mt-2 pl-6">
                          {folder.files.length > 0 ? (
                            <ul className="list-disc">
                              {folder.files.map((file, fileIndex) => (
                                <li key={fileIndex} className="flex items-center justify-between">
                                  <span
                                    className="cursor-pointer text-blue-500 hover:underline"
                                    onClick={() => handleFileSelect(folder.name, file)}
                                  >
                                    {file}
                                  </span>
                                  <button
                                    onClick={() => handleDeleteFile(folder.name, file)}
                                    className="text-gray-800 hover:text-gray-700"
                                  >
                                    <TrashIcon className="h-4 w-4" />
                                  </button>
                                </li>
                              ))}
                            </ul>
                          ) : (
                            <p className="text-gray-600">No files yet</p>
                          )}
                        </div>
                      )}
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          </div>

          {/* File Content */}
          <div className="w-1/2 p-2 h-full">
            <div className="h-full border border-gray-300 rounded-lg p-2 flex flex-col" ref={fileContentContainerRef}>
              <div className="flex-grow relative flex justify-center" ref={fileContentSubContainerRef}>
                <canvas
                  ref={fileContentCanvasRef}
                  className="absolute top-0 w-auto h-full object-contain"
                  width={fileDimensions.width}
                  height={fileDimensions.height}
                />
                <p className="absolute top-1 left-1 text-m text-white bg-black bg-opacity-50 px-2 py-1 rounded">File Content</p>
                <div className="absolute bottom-1 right-1 left-1 flex items-center space-x-1">
                  <button
                    className="bg-gray-800 p-3 rounded-full hover:bg-gray-700 focus:outline-none"
                    onClick={isPlayingFile ? handleFilePause : handleFilePlay}
                    disabled={!fileContent}
                  >
                    {isPlayingFile ? (
                      <PauseIcon className="h-6 w-6 text-white" />
                    ) : (
                      <PlayIcon className="h-6 w-6 text-white" />
                    )}
                  </button>
                  <input
                    type="range"
                    min="0"
                    max={fileContent ? fileContent.length - 1 : 0}
                    value={currentFileFrame}
                    onChange={handleFileSeek}
                    className="flex-grow accent-gray-800 hover:accent-gray-700"
                  />
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Video;