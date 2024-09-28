import './App.css';
import React from 'react';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import Home from './components/Home.jsx';
import Room from './components/Room.jsx';
import Video from './components/Video.jsx';


function App() {
  return (
    <Router>
    <Routes>
      <Route path="/" element={<Home />} />
      <Route path="/room/:roomId/:initiator/" element={<Room />} />
      <Route path="/collect/" element={<Video />} />
    </Routes>
  </Router>
  );
}

export default App;
