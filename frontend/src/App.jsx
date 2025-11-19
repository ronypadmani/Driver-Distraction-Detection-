import React, { useState } from 'react';
import VideoUpload from './components/VideoUpload';
import ProcessingStatus from './components/ProcessingStatus';
import ResultsDisplay from './components/ResultsDisplay';
import Header from './components/Header';
import './App.css';

function App() {
  const [currentStep, setCurrentStep] = useState('upload'); // 'upload', 'processing', 'results'
  const [uploadedFile, setUploadedFile] = useState(null);
  const [processingStatus, setProcessingStatus] = useState('');
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);

  const handleFileUpload = (file) => {
    setUploadedFile(file);
    setCurrentStep('processing');
    setError(null);
    processVideo(file);
  };

  const processVideo = async (file) => {
    try {
      setProcessingStatus('Uploading video...');
      
      const formData = new FormData();
      formData.append('video', file);

      // Create XMLHttpRequest for progress tracking
      const xhr = new XMLHttpRequest();
      
      xhr.upload.addEventListener('progress', (e) => {
        if (e.lengthComputable) {
          const percentComplete = (e.loaded / e.total) * 100;
          setProcessingStatus(`Uploading video... ${Math.round(percentComplete)}%`);
        }
      });

      xhr.addEventListener('load', () => {
        if (xhr.status === 200) {
          const response = JSON.parse(xhr.responseText);
          setResults(response);
          setCurrentStep('results');
          setProcessingStatus('');
        } else {
          const errorResponse = JSON.parse(xhr.responseText);
          setError(errorResponse.error || 'Processing failed');
          setCurrentStep('upload');
          setProcessingStatus('');
        }
      });

      xhr.addEventListener('error', () => {
        setError('Network error occurred');
        setCurrentStep('upload');
        setProcessingStatus('');
      });

      setProcessingStatus('Processing video...');
      xhr.open('POST', '/predict_video');
      xhr.send(formData);

    } catch (err) {
      setError('Failed to process video: ' + err.message);
      setCurrentStep('upload');
      setProcessingStatus('');
    }
  };

  const handleReset = () => {
    setCurrentStep('upload');
    setUploadedFile(null);
    setProcessingStatus('');
    setResults(null);
    setError(null);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      <Header />
      
      <main className="container mx-auto px-4 py-8">
        {error && (
          <div className="mb-6 bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded-lg">
            <div className="flex items-center">
              <svg className="w-5 h-5 mr-2" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
              </svg>
              <span className="font-medium">Error: </span>
              {error}
            </div>
            <button 
              onClick={handleReset}
              className="mt-2 text-sm underline hover:no-underline"
            >
              Try again
            </button>
          </div>
        )}

        {currentStep === 'upload' && (
          <VideoUpload onFileUpload={handleFileUpload} />
        )}

        {currentStep === 'processing' && (
          <ProcessingStatus 
            status={processingStatus}
            fileName={uploadedFile?.name}
          />
        )}

        {currentStep === 'results' && results && (
          <ResultsDisplay 
            results={results}
            onReset={handleReset}
            fileName={uploadedFile?.name}
          />
        )}
      </main>

      <footer className="bg-white border-t border-gray-200 py-6 mt-12">
        <div className="container mx-auto px-4 text-center text-gray-600">
          <p>Driver Distraction Detection System</p>
          <p className="text-sm mt-1">Powered by Computer Vision & Machine Learning</p>
        </div>
      </footer>
    </div>
  );
}

export default App;
