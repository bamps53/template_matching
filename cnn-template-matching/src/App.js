import React, { useState } from 'react';
import axios from 'axios';

function App() {
  const [selectedFile, setSelectedFile] = useState();
  const [selectedTemplate, setSelectedTemplate] = useState();
  const [resultImage, setResultImage] = useState();

  const submitImage = async () => {
    const formData = new FormData();
    formData.append('file', selectedFile);

    const result = await axios.post('http://localhost:8000/upload_image/', formData, {
      headers: {'Content-Type': 'multipart/form-data'}
    });

    setResultImage("data:image/jpeg;base64," + result.data.image);
  };

  const submitTemplate = async () => {
    const formData = new FormData();
    formData.append('file', selectedTemplate);

    await axios.post('http://localhost:8000/upload_template/', formData, {
      headers: {'Content-Type': 'multipart/form-data'}
    });
  };

  return (
    <div className="App">
      <h3>Upload Template</h3>
      <input type="file" onChange={event => setSelectedTemplate(event.target.files[0])} />
      <button onClick={submitTemplate}>Submit Template</button>

      <h3>Upload Image</h3>
      <input type="file" onChange={event => setSelectedFile(event.target.files[0])} />
      <button onClick={submitImage}>Submit Image</button>

      {resultImage && <img src={resultImage} alt="Result" />}
    </div>
  );
}

export default App;
