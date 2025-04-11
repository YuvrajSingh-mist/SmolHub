import React from 'react';
import { Link } from 'react-router-dom';

function Home() {
  return (
    <div>
      <h1>SmolHub</h1>
      <div>
        <Link to="/models">Models</Link>
        <Link to="/datasets">Datasets</Link>
      </div>
    </div>
  );
}

export default Home;
