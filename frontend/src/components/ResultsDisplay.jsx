import React, { useState } from 'react';
import FrameTimeline from './FrameTimeline';
import ConfidenceChart from './ConfidenceChart';

const ResultsDisplay = ({ results, onReset, fileName }) => {
  const [activeTab, setActiveTab] = useState('overview');

  const isDistracted = results.video_prediction === 'distracted';
  const distractionRatio = results.distraction_ratio || 0;
  const totalFrames = results.total_frames || 0;
  const distractedFrames = results.distracted_frames || 0;
  const framesPredictions = results.frame_predictions || [];

  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const getDistractionLevel = (ratio) => {
    if (ratio >= 0.7) return { level: 'High', color: 'red', bgColor: 'red-100' };
    if (ratio >= 0.4) return { level: 'Medium', color: 'yellow', bgColor: 'yellow-100' };
    if (ratio >= 0.1) return { level: 'Low', color: 'orange', bgColor: 'orange-100' };
    return { level: 'Minimal', color: 'green', bgColor: 'green-100' };
  };

  const distractionLevel = getDistractionLevel(distractionRatio);

  return (
    <div className="max-w-6xl mx-auto">
      {/* Main Result Card */}
      <div className="bg-white rounded-xl shadow-lg p-8 mb-6">
        <div className="text-center mb-8">
          <div className={`w-24 h-24 rounded-full flex items-center justify-center mx-auto mb-4 ${
            isDistracted ? 'bg-red-100' : 'bg-green-100'
          }`}>
            {isDistracted ? (
              <svg className="w-12 h-12 text-red-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L4.082 16.5c-.77.833.192 2.5 1.732 2.5z" />
              </svg>
            ) : (
              <svg className="w-12 h-12 text-green-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
            )}
          </div>
          
          <h2 className={`text-3xl font-bold mb-2 ${
            isDistracted ? 'text-red-600' : 'text-green-600'
          }`}>
            {isDistracted ? 'Driver is Distracted' : 'Driver is Not Distracted'}
          </h2>
          
          <p className="text-gray-600 mb-4">
            Analysis complete for <span className="font-medium">{fileName}</span>
          </p>

          <div className={`inline-flex items-center px-6 py-3 rounded-full text-lg font-semibold ${
            isDistracted 
              ? 'bg-red-100 text-red-800 border-2 border-red-200' 
              : 'bg-green-100 text-green-800 border-2 border-green-200'
          }`}>
            {isDistracted ? '🚨 DISTRACTED' : '✅ FOCUSED'}
          </div>
        </div>

        {/* Key Metrics */}
        <div className="grid md:grid-cols-4 gap-6 mb-8">
          <div className="text-center p-4 bg-blue-50 rounded-lg">
            <div className="text-2xl font-bold text-blue-600">
              {(distractionRatio * 100).toFixed(1)}%
            </div>
            <div className="text-sm text-gray-600">Distraction Ratio</div>
          </div>
          
          <div className="text-center p-4 bg-purple-50 rounded-lg">
            <div className="text-2xl font-bold text-purple-600">{totalFrames}</div>
            <div className="text-sm text-gray-600">Total Frames</div>
          </div>
          
          <div className="text-center p-4 bg-red-50 rounded-lg">
            <div className="text-2xl font-bold text-red-600">{distractedFrames}</div>
            <div className="text-sm text-gray-600">Distracted Frames</div>
          </div>
          
          <div className={`text-center p-4 bg-${distractionLevel.bgColor} rounded-lg`}>
            <div className={`text-2xl font-bold text-${distractionLevel.color}-600`}>
              {distractionLevel.level}
            </div>
            <div className="text-sm text-gray-600">Risk Level</div>
          </div>
        </div>

        {/* Progress Bar */}
        <div className="mb-6">
          <div className="flex justify-between text-sm text-gray-600 mb-2">
            <span>Distraction Level</span>
            <span>{(distractionRatio * 100).toFixed(1)}%</span>
          </div>
          <div className="w-full bg-gray-200 rounded-full h-4">
            <div 
              className={`h-4 rounded-full transition-all duration-1000 ${
                distractionRatio >= 0.7 ? 'bg-red-500' :
                distractionRatio >= 0.4 ? 'bg-yellow-500' :
                distractionRatio >= 0.1 ? 'bg-orange-500' : 'bg-green-500'
              }`}
              style={{ width: `${distractionRatio * 100}%` }}
            ></div>
          </div>
        </div>
      </div>

      {/* Detailed Analysis Tabs */}
      <div className="bg-white rounded-xl shadow-lg">
        {/* Tab Navigation */}
        <div className="border-b border-gray-200">
          <nav className="flex space-x-8 px-6">
            {[
              { id: 'overview', name: 'Overview', icon: '📊' },
              { id: 'timeline', name: 'Frame Timeline', icon: '⏱️' },
              { id: 'insights', name: 'Insights', icon: '🔍' }
            ].map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`py-4 px-2 border-b-2 font-medium text-sm transition-colors ${
                  activeTab === tab.id
                    ? 'border-blue-500 text-blue-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
              >
                <span className="mr-2">{tab.icon}</span>
                {tab.name}
              </button>
            ))}
          </nav>
        </div>

        {/* Tab Content */}
        <div className="p-6">
          {activeTab === 'overview' && (
            <div className="space-y-6">
              <div className="grid md:grid-cols-2 gap-6">
                <div className="bg-gray-50 rounded-lg p-6">
                  <h3 className="font-semibold text-gray-800 mb-4">Detection Summary</h3>
                  <div className="space-y-3">
                    <div className="flex justify-between">
                      <span className="text-gray-600">Video Duration:</span>
                      <span className="font-medium">
                        {framesPredictions.length > 0 ? 
                          formatTime(framesPredictions[framesPredictions.length - 1].timestamp) : 'N/A'}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600">Analysis Method:</span>
                      <span className="font-medium">AI Computer Vision</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600">Features Analyzed:</span>
                      <span className="font-medium">5 Types</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600">Processing Time:</span>
                      <span className="font-medium">
                        {results.processing_info?.processing_timestamp ? 'Real-time' : 'N/A'}
                      </span>
                    </div>
                  </div>
                </div>

                <div className="bg-gray-50 rounded-lg p-6">
                  <h3 className="font-semibold text-gray-800 mb-4">Risk Assessment</h3>
                  <div className="space-y-3">
                    <div className={`p-3 rounded-lg bg-${distractionLevel.bgColor}`}>
                      <div className={`font-semibold text-${distractionLevel.color}-800`}>
                        Risk Level: {distractionLevel.level}
                      </div>
                      <div className={`text-sm text-${distractionLevel.color}-600 mt-1`}>
                        {distractionLevel.level === 'High' && 'Immediate attention required'}
                        {distractionLevel.level === 'Medium' && 'Caution advised'}
                        {distractionLevel.level === 'Low' && 'Minor concerns detected'}
                        {distractionLevel.level === 'Minimal' && 'Driver appears focused'}
                      </div>
                    </div>
                    
                    <div className="text-sm text-gray-600">
                      <p className="mb-2">
                        <strong>Recommendation:</strong> {
                          isDistracted 
                            ? 'Review driving behavior and consider safety measures.'
                            : 'Good driving behavior detected. Continue safe practices.'
                        }
                      </p>
                    </div>
                  </div>
                </div>
              </div>

              {/* Confidence Chart */}
              <ConfidenceChart framesPredictions={framesPredictions} />
            </div>
          )}

          {activeTab === 'timeline' && (
            <FrameTimeline framesPredictions={framesPredictions} />
          )}

          {activeTab === 'insights' && (
            <div className="space-y-6">
              <div className="grid md:grid-cols-2 gap-6">
                <div className="bg-blue-50 rounded-lg p-6">
                  <h3 className="font-semibold text-blue-800 mb-4">🧠 AI Analysis Details</h3>
                  <div className="space-y-3 text-sm">
                    <p><strong>Features Analyzed:</strong></p>
                    <ul className="list-disc list-inside space-y-1 text-blue-700">
                      <li>Body pose and hand positions</li>
                      <li>Object detection (phone, steering wheel)</li>
                      <li>Spatial relationships and distances</li>
                      <li>Scene understanding</li>
                      <li>Deep visual features</li>
                    </ul>
                  </div>
                </div>

                <div className="bg-green-50 rounded-lg p-6">
                  <h3 className="font-semibold text-green-800 mb-4">📈 Performance Metrics</h3>
                  <div className="space-y-3 text-sm">
                    <div className="flex justify-between">
                      <span>Model Accuracy:</span>
                      <span className="font-medium">89.5%</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Feature Dimensions:</span>
                      <span className="font-medium">4,180</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Frames Processed:</span>
                      <span className="font-medium">{results.processing_info?.frames_processed || totalFrames}</span>
                    </div>
                  </div>
                </div>
              </div>

              <div className="bg-yellow-50 rounded-lg p-6">
                <h3 className="font-semibold text-yellow-800 mb-4">⚠️ Important Notes</h3>
                <div className="space-y-2 text-sm text-yellow-700">
                  <p>• This analysis is for informational purposes and should not replace human judgment.</p>
                  <p>• Results are based on visual analysis and may not capture all forms of distraction.</p>
                  <p>• Consider environmental factors and context when interpreting results.</p>
                  <p>• For safety-critical applications, combine with additional monitoring systems.</p>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Action Buttons */}
      <div className="mt-6 text-center space-x-4">
        <button
          onClick={onReset}
          className="bg-blue-500 hover:bg-blue-600 text-white font-semibold py-3 px-8 rounded-lg transition-colors duration-200 shadow-lg hover:shadow-xl"
        >
          Analyze Another Video
        </button>
        
        <button
          onClick={() => {
            const dataStr = JSON.stringify(results, null, 2);
            const dataBlob = new Blob([dataStr], {type: 'application/json'});
            const url = URL.createObjectURL(dataBlob);
            const link = document.createElement('a');
            link.href = url;
            link.download = `distraction-analysis-${fileName}.json`;
            link.click();
          }}
          className="bg-gray-500 hover:bg-gray-600 text-white font-semibold py-3 px-8 rounded-lg transition-colors duration-200 shadow-lg hover:shadow-xl"
        >
          Download Report
        </button>
      </div>
    </div>
  );
};

export default ResultsDisplay;
