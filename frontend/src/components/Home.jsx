import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import logo from '../logo.png';

const Home = () => {
  const [roomId, setRoomId] = useState('');
  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();

    try {
      // Make a POST request to the Django backend
      const response = await fetch('http://localhost:8000/room_info/', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ id: roomId }),
      });

      if (response.ok) {
        const data = await response.json();
        navigate(`/room/${roomId}/${data.message}/`);
      } else {
        console.error('Failed to create/join room', response.statusText);
      }
    } catch (error) {
      console.error('Error:', error);
    }
  };

  return (
    <div className="flex items-center justify-center min-h-screen bg-gray-100">
      <div className="w-full max-w-md p-8 bg-white rounded-lg shadow-md">
        <div className="text-center mb-8">
          <img src={logo} alt="Logo" className="mx-auto mb-4 h-20" />
          <h1 className="text-2xl font-bold text-gray-800">Sign Language Video Call App</h1>
          <p className="text-gray-600">Join a room or create a custom pose</p>
        </div>
        <div>
          <div>
            <form onSubmit={handleSubmit} className="mb-6">
              <div className="mb-4">
                <input
                  type="text"
                  value={roomId}
                  onChange={(e) => setRoomId(e.target.value)}
                  placeholder="Enter Room ID"
                  className="w-full px-4 py-2 text-lg border rounded-md border-gray-300 focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
              </div>
              <button
                type="submit"
                className="w-full px-4 py-2 text-lg text-white bg-blue-500 rounded-md hover:bg-blue-600 focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                Join Room
              </button>
            </form>
          </div>
          <hr class="h-px my-8 bg-gray-200 border-0 dark:bg-gray-700"/>
          <div className="text-center">
            <button
              onClick={() => navigate('/collect')}
              className="w-full px-4 py-2 text-lg text-white bg-indigo-500 rounded-md hover:bg-indigo-600 focus:outline-none focus:ring-2 focus:ring-yellow-500"
            >
              Create Custom Pose
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Home;
