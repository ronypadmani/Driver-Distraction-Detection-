import React, { useState, useEffect } from 'react';

const ProcessingStatus = ({ status, fileName }) => {
  const [currentStep, setCurrentStep] = useState(0);
  const [dots, setDots] = useState('');

  const steps = [
    { name: 'Uploading video', icon: '📤', description: 'Transferring your video file' },
    { name: 'Extracting frames', icon: '🎬', description: 'Breaking video into individual frames' },
    { name: 'Analyzing poses', icon: '🤸', description: 'Detecting body and hand positions' },
    { name: 'Detecting objects', icon: '🔍', description: 'Identifying objects in the scene' },
    { name: 'Processing features', icon: '🧠', description: 'Extracting AI features' },
    { name: 'Making predictions', icon: '⚡', description: 'Running distraction detection' }
  ];

  useEffect(() => {
    // Animate dots
    const interval = setInterval(() => {
      setDots(prev => prev.length >= 3 ? '' : prev + '.');
    }, 500);

    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    // Update current step based on status
    if (status.includes('Uploading')) {
      setCurrentStep(0);
    } else if (status.includes('Processing') || status.includes('Extracting')) {
      setCurrentStep(1);
    } else if (status.includes('Analyzing') || status.includes('pose')) {
      setCurrentStep(2);
    } else if (status.includes('Detecting') || status.includes('object')) {
      setCurrentStep(3);
    } else if (status.includes('features')) {
      setCurrentStep(4);
    } else if (status.includes('prediction')) {
      setCurrentStep(5);
    }
  }, [status]);

  return (
    <div className="max-w-4xl mx-auto">
      <div className="bg-white rounded-xl shadow-lg p-8">
        {/* Header */}
        <div className="text-center mb-8">
          <div className="w-20 h-20 bg-blue-100 rounded-full flex items-center justify-center mx-auto mb-4 animate-pulse">
            <svg className="w-10 h-10 text-blue-500 animate-spin" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
            </svg>
          </div>
          <h2 className="text-2xl font-bold text-gray-800 mb-2">
            Processing Your Video
          </h2>
          <p className="text-gray-600">
            Analyzing <span className="font-medium text-blue-600">{fileName}</span> for driver distraction
          </p>
        </div>

        {/* Current Status */}
        <div className="bg-blue-50 rounded-lg p-6 mb-8 text-center">
          <div className="text-lg font-semibold text-blue-800 mb-2">
            {status}{dots}
          </div>
          <div className="w-full bg-blue-200 rounded-full h-2">
            <div 
              className="bg-blue-500 h-2 rounded-full transition-all duration-1000 ease-out"
              style={{ width: `${((currentStep + 1) / steps.length) * 100}%` }}
            ></div>
          </div>
          <div className="text-sm text-blue-600 mt-2">
            Step {currentStep + 1} of {steps.length}
          </div>
        </div>

        {/* Processing Steps */}
        <div className="space-y-4">
          {steps.map((step, index) => (
            <div 
              key={index}
              className={`flex items-center p-4 rounded-lg transition-all duration-300 ${
                index <= currentStep 
                  ? 'bg-green-50 border-l-4 border-green-500' 
                  : index === currentStep + 1
                  ? 'bg-blue-50 border-l-4 border-blue-500 animate-pulse'
                  : 'bg-gray-50 border-l-4 border-gray-300'
              }`}
            >
              <div className="text-2xl mr-4">
                {index < currentStep ? '✅' : index === currentStep ? step.icon : '⏳'}
              </div>
              <div className="flex-1">
                <div className={`font-semibold ${
                  index <= currentStep ? 'text-green-800' : 
                  index === currentStep + 1 ? 'text-blue-800' : 'text-gray-600'
                }`}>
                  {step.name}
                </div>
                <div className={`text-sm ${
                  index <= currentStep ? 'text-green-600' : 
                  index === currentStep + 1 ? 'text-blue-600' : 'text-gray-500'
                }`}>
                  {step.description}
                </div>
              </div>
              {index <= currentStep && (
                <div className="text-green-500">
                  <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                  </svg>
                </div>
              )}
            </div>
          ))}
        </div>

        {/* Processing Info */}
        <div className="mt-8 bg-gray-50 rounded-lg p-6">
          <h3 className="font-semibold text-gray-800 mb-3">What's happening?</h3>
          <div className="grid md:grid-cols-2 gap-4 text-sm text-gray-600">
            <div>
              <p className="mb-2">
                <span className="font-medium">🎯 AI Analysis:</span> Our system uses advanced computer vision to detect driver behavior patterns.
              </p>
              <p className="mb-2">
                <span className="font-medium">🔍 Multi-Feature Detection:</span> We analyze pose, objects, distances, and visual features.
              </p>
            </div>
            <div>
              <p className="mb-2">
                <span className="font-medium">⚡ Real-time Processing:</span> Each frame is processed through our trained neural network.
              </p>
              <p className="mb-2">
                <span className="font-medium">📊 Comprehensive Results:</span> You'll get frame-by-frame analysis and overall assessment.
              </p>
            </div>
          </div>
        </div>

        {/* Estimated Time */}
        <div className="mt-6 text-center">
          <div className="inline-flex items-center px-4 py-2 bg-yellow-100 rounded-full">
            <svg className="w-4 h-4 text-yellow-600 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            <span className="text-sm text-yellow-800">
              Processing typically takes 30-60 seconds
            </span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ProcessingStatus;
