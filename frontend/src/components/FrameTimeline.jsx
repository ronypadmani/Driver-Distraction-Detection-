import React, { useState } from 'react';

const FrameTimeline = ({ framesPredictions }) => {
  const [selectedFrame, setSelectedFrame] = useState(null);
  const [currentPage, setCurrentPage] = useState(0);
  const framesPerPage = 20;

  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const getFrameColor = (prediction, confidence) => {
    if (prediction === 1) {
      return confidence > 0.8 ? 'bg-red-500' : confidence > 0.6 ? 'bg-red-400' : 'bg-red-300';
    } else {
      return confidence > 0.8 ? 'bg-green-500' : confidence > 0.6 ? 'bg-green-400' : 'bg-green-300';
    }
  };

  const totalPages = Math.ceil(framesPredictions.length / framesPerPage);
  const currentFrames = framesPredictions.slice(
    currentPage * framesPerPage,
    (currentPage + 1) * framesPerPage
  );

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h3 className="text-xl font-semibold text-gray-800">Frame-by-Frame Analysis</h3>
        <div className="text-sm text-gray-600">
          {framesPredictions.length} frames analyzed
        </div>
      </div>

      {/* Legend */}
      <div className="bg-gray-50 rounded-lg p-4">
        <div className="flex items-center justify-center space-x-8 text-sm">
          <div className="flex items-center space-x-2">
            <div className="w-4 h-4 bg-red-500 rounded"></div>
            <span>Distracted (High Confidence)</span>
          </div>
          <div className="flex items-center space-x-2">
            <div className="w-4 h-4 bg-red-300 rounded"></div>
            <span>Distracted (Low Confidence)</span>
          </div>
          <div className="flex items-center space-x-2">
            <div className="w-4 h-4 bg-green-500 rounded"></div>
            <span>Focused (High Confidence)</span>
          </div>
          <div className="flex items-center space-x-2">
            <div className="w-4 h-4 bg-green-300 rounded"></div>
            <span>Focused (Low Confidence)</span>
          </div>
        </div>
      </div>

      {/* Timeline Visualization */}
      <div className="bg-white border rounded-lg p-6">
        <div className="mb-4">
          <h4 className="font-medium text-gray-800 mb-2">Timeline Overview</h4>
          <div className="flex space-x-1 h-8 bg-gray-100 rounded overflow-hidden">
            {framesPredictions.map((frame, index) => (
              <div
                key={index}
                className={`flex-1 cursor-pointer transition-all duration-200 hover:opacity-80 ${
                  getFrameColor(frame.prediction, frame.confidence)
                }`}
                onClick={() => setSelectedFrame(frame)}
                title={`Frame ${index + 1}: ${frame.label} (${(frame.confidence * 100).toFixed(1)}%)`}
              />
            ))}
          </div>
          <div className="flex justify-between text-xs text-gray-500 mt-1">
            <span>0:00</span>
            <span>
              {framesPredictions.length > 0 ? 
                formatTime(framesPredictions[framesPredictions.length - 1].timestamp) : '0:00'}
            </span>
          </div>
        </div>
      </div>

      {/* Frame Details Grid */}
      <div className="grid gap-4">
        <div className="flex justify-between items-center">
          <h4 className="font-medium text-gray-800">Detailed Frame Analysis</h4>
          {totalPages > 1 && (
            <div className="flex items-center space-x-2">
              <button
                onClick={() => setCurrentPage(Math.max(0, currentPage - 1))}
                disabled={currentPage === 0}
                className="px-3 py-1 bg-gray-200 rounded disabled:opacity-50 hover:bg-gray-300 transition-colors"
              >
                Previous
              </button>
              <span className="text-sm text-gray-600">
                Page {currentPage + 1} of {totalPages}
              </span>
              <button
                onClick={() => setCurrentPage(Math.min(totalPages - 1, currentPage + 1))}
                disabled={currentPage === totalPages - 1}
                className="px-3 py-1 bg-gray-200 rounded disabled:opacity-50 hover:bg-gray-300 transition-colors"
              >
                Next
              </button>
            </div>
          )}
        </div>

        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-3">
          {currentFrames.map((frame, index) => (
            <div
              key={currentPage * framesPerPage + index}
              className={`border rounded-lg p-3 cursor-pointer transition-all duration-200 hover:shadow-md ${
                selectedFrame === frame ? 'ring-2 ring-blue-500 bg-blue-50' : 'bg-white hover:bg-gray-50'
              }`}
              onClick={() => setSelectedFrame(frame)}
            >
              <div className="flex items-center justify-between mb-2">
                <div className="text-sm font-medium text-gray-800">
                  Frame {currentPage * framesPerPage + index + 1}
                </div>
                <div className={`w-3 h-3 rounded-full ${
                  getFrameColor(frame.prediction, frame.confidence)
                }`}></div>
              </div>
              
              <div className="text-xs text-gray-600 space-y-1">
                <div>Time: {formatTime(frame.timestamp)}</div>
                <div className={`font-medium ${
                  frame.prediction === 1 ? 'text-red-600' : 'text-green-600'
                }`}>
                  {frame.label}
                </div>
                <div>Confidence: {(frame.confidence * 100).toFixed(1)}%</div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Selected Frame Details */}
      {selectedFrame && (
        <div className="bg-white border rounded-lg p-6">
          <h4 className="font-medium text-gray-800 mb-4">Selected Frame Details</h4>
          <div className="grid md:grid-cols-2 gap-6">
            <div className="space-y-3">
              <div className="flex justify-between">
                <span className="text-gray-600">Frame:</span>
                <span className="font-medium">{selectedFrame.frame}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Timestamp:</span>
                <span className="font-medium">{formatTime(selectedFrame.timestamp)}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Prediction:</span>
                <span className={`font-medium ${
                  selectedFrame.prediction === 1 ? 'text-red-600' : 'text-green-600'
                }`}>
                  {selectedFrame.label}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Confidence:</span>
                <span className="font-medium">{(selectedFrame.confidence * 100).toFixed(1)}%</span>
              </div>
            </div>
            
            <div className="space-y-3">
              <div className="text-sm text-gray-600">
                <div className="font-medium mb-2">Analysis Details:</div>
                <div className="bg-gray-50 rounded p-3 space-y-1">
                  <div>• Pose detection: Active</div>
                  <div>• Object recognition: Active</div>
                  <div>• Distance calculation: Active</div>
                  <div>• Feature extraction: Complete</div>
                </div>
              </div>
            </div>
          </div>
          
          {/* Confidence Bar */}
          <div className="mt-4">
            <div className="flex justify-between text-sm text-gray-600 mb-1">
              <span>Confidence Level</span>
              <span>{(selectedFrame.confidence * 100).toFixed(1)}%</span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-3">
              <div 
                className={`h-3 rounded-full transition-all duration-300 ${
                  selectedFrame.prediction === 1 ? 'bg-red-500' : 'bg-green-500'
                }`}
                style={{ width: `${selectedFrame.confidence * 100}%` }}
              ></div>
            </div>
          </div>
        </div>
      )}

      {/* Statistics Summary */}
      <div className="bg-gray-50 rounded-lg p-6">
        <h4 className="font-medium text-gray-800 mb-4">Frame Statistics</h4>
        <div className="grid md:grid-cols-4 gap-4 text-center">
          <div className="bg-white rounded p-4">
            <div className="text-2xl font-bold text-green-600">
              {framesPredictions.filter(f => f.prediction === 0).length}
            </div>
            <div className="text-sm text-gray-600">Focused Frames</div>
          </div>
          <div className="bg-white rounded p-4">
            <div className="text-2xl font-bold text-red-600">
              {framesPredictions.filter(f => f.prediction === 1).length}
            </div>
            <div className="text-sm text-gray-600">Distracted Frames</div>
          </div>
          <div className="bg-white rounded p-4">
            <div className="text-2xl font-bold text-blue-600">
              {framesPredictions.length > 0 ? 
                (framesPredictions.reduce((sum, f) => sum + f.confidence, 0) / framesPredictions.length * 100).toFixed(1) + '%'
                : '0%'}
            </div>
            <div className="text-sm text-gray-600">Avg Confidence</div>
          </div>
          <div className="bg-white rounded p-4">
            <div className="text-2xl font-bold text-purple-600">
              {framesPredictions.filter(f => f.confidence > 0.8).length}
            </div>
            <div className="text-sm text-gray-600">High Confidence</div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default FrameTimeline;
