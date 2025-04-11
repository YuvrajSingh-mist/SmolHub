import React, { useState, useEffect } from 'react';
import axios from 'axios';

function ModelList() {
  const [models, setModels] = useState([]);

  useEffect(() => {
    const fetchModels = async () => {
      const response = await axios.get('http://localhost:8000/api/hub/models/');
      setModels(response.data);
    };
    fetchModels();
  }, []);

  return (
    <div>
      <h2>Available Models</h2>
      {models.map(model => (
        <div key={model.id}>
          <h3>{model.name}</h3>
          <p>{model.description}</p>
          <button onClick={() => window.location.href = model.file}>Download</button>
        </div>
      ))}
    </div>
  );
}

export default ModelList;
