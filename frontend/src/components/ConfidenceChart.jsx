import React from 'react';

const ConfidenceChart = ({ framesPredictions }) => {
  if (!framesPredictions || framesPredictions.length === 0) {
    return (
      <div className="bg-gray-50 rounded-lg p-6 text-center">
        <p className="text-gray-500">No frame data available for chart</p>
      </div>
    );
  }

  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  // Calculate chart dimensions
  const chartWidth = 800;
  const chartHeight = 200;
  const padding = { top: 20, right: 40, bottom: 40, left: 60 };
  const innerWidth = chartWidth - padding.left - padding.right;
  const innerHeight = chartHeight - padding.top - padding.bottom;

  // Create data points
  const maxTime = framesPredictions[framesPredictions.length - 1]?.timestamp || 1;
  const points = framesPredictions.map((frame, index) => ({
    x: padding.left + (frame.timestamp / maxTime) * innerWidth,
    y: padding.top + (1 - frame.confidence) * innerHeight,
    frame: frame,
    index: index
  }));

  // Create path for line chart
  const pathData = points.map((point, index) => 
    `${index === 0 ? 'M' : 'L'} ${point.x} ${point.y}`
  ).join(' ');

  // Create area fill path
  const areaPath = `M ${padding.left} ${padding.top + innerHeight} L ${points.map(p => `${p.x} ${p.y}`).join(' L ')} L ${padding.left + innerWidth} ${padding.top + innerHeight} Z`;

  return (
    <div className="bg-white rounded-lg p-6 border">
      <h3 className="font-semibold text-gray-800 mb-4">Confidence Over Time</h3>
      
      <div className="overflow-x-auto">
        <svg width={chartWidth} height={chartHeight} className="border rounded">
          {/* Grid lines */}
          <defs>
            <pattern id="grid" width="40" height="20" patternUnits="userSpaceOnUse">
              <path d="M 40 0 L 0 0 0 20" fill="none" stroke="#f0f0f0" strokeWidth="1"/>
            </pattern>
          </defs>
          <rect width={chartWidth} height={chartHeight} fill="url(#grid)" />
          
          {/* Y-axis labels */}
          {[0, 0.25, 0.5, 0.75, 1].map(value => (
            <g key={value}>
              <line 
                x1={padding.left - 5} 
                y1={padding.top + (1 - value) * innerHeight}
                x2={padding.left} 
                y2={padding.top + (1 - value) * innerHeight}
                stroke="#666" 
                strokeWidth="1"
              />
              <text 
                x={padding.left - 10} 
                y={padding.top + (1 - value) * innerHeight + 4}
                textAnchor="end" 
                fontSize="12" 
                fill="#666"
              >
                {(value * 100).toFixed(0)}%
              </text>
            </g>
          ))}
          
          {/* X-axis labels */}
          {[0, 0.25, 0.5, 0.75, 1].map(ratio => {
            const time = maxTime * ratio;
            return (
              <g key={ratio}>
                <line 
                  x1={padding.left + ratio * innerWidth} 
                  y1={padding.top + innerHeight}
                  x2={padding.left + ratio * innerWidth} 
                  y2={padding.top + innerHeight + 5}
                  stroke="#666" 
                  strokeWidth="1"
                />
                <text 
                  x={padding.left + ratio * innerWidth} 
                  y={padding.top + innerHeight + 20}
                  textAnchor="middle" 
                  fontSize="12" 
                  fill="#666"
                >
                  {formatTime(time)}
                </text>
              </g>
            );
          })}
          
          {/* Area fill */}
          <path 
            d={areaPath} 
            fill="rgba(59, 130, 246, 0.1)" 
            stroke="none"
          />
          
          {/* Main line */}
          <path 
            d={pathData} 
            fill="none" 
            stroke="#3b82f6" 
            strokeWidth="2"
          />
          
          {/* Data points */}
          {points.map((point, index) => (
            <g key={index}>
              <circle 
                cx={point.x} 
                cy={point.y} 
                r="4" 
                fill={point.frame.prediction === 1 ? "#ef4444" : "#22c55e"}
                stroke="white" 
                strokeWidth="2"
                className="hover:r-6 cursor-pointer transition-all"
              >
                <title>
                  Frame {index + 1}: {point.frame.label} ({(point.frame.confidence * 100).toFixed(1)}%)
                </title>
              </circle>
            </g>
          ))}
          
          {/* Axis labels */}
          <text 
            x={chartWidth / 2} 
            y={chartHeight - 5} 
            textAnchor="middle" 
            fontSize="14" 
            fill="#374151" 
            fontWeight="500"
          >
            Time
          </text>
          <text 
            x={15} 
            y={chartHeight / 2} 
            textAnchor="middle" 
            fontSize="14" 
            fill="#374151" 
            fontWeight="500"
            transform={`rotate(-90, 15, ${chartHeight / 2})`}
          >
            Confidence
          </text>
        </svg>
      </div>

      {/* Chart Legend */}
      <div className="flex justify-center items-center space-x-6 mt-4 text-sm">
        <div className="flex items-center space-x-2">
          <div className="w-4 h-4 bg-green-500 rounded-full"></div>
          <span>Not Distracted</span>
        </div>
        <div className="flex items-center space-x-2">
          <div className="w-4 h-4 bg-red-500 rounded-full"></div>
          <span>Distracted</span>
        </div>
        <div className="flex items-center space-x-2">
          <div className="w-4 h-1 bg-blue-500"></div>
          <span>Confidence Trend</span>
        </div>
      </div>

      {/* Summary Statistics */}
      <div className="grid md:grid-cols-3 gap-4 mt-6 pt-4 border-t">
        <div className="text-center">
          <div className="text-lg font-semibold text-blue-600">
            {framesPredictions.length > 0 ? 
              (framesPredictions.reduce((sum, f) => sum + f.confidence, 0) / framesPredictions.length * 100).toFixed(1) + '%'
              : '0%'}
          </div>
          <div className="text-sm text-gray-600">Average Confidence</div>
        </div>
        <div className="text-center">
          <div className="text-lg font-semibold text-green-600">
            {framesPredictions.length > 0 ? 
              (Math.max(...framesPredictions.map(f => f.confidence)) * 100).toFixed(1) + '%'
              : '0%'}
          </div>
          <div className="text-sm text-gray-600">Highest Confidence</div>
        </div>
        <div className="text-center">
          <div className="text-lg font-semibold text-orange-600">
            {framesPredictions.length > 0 ? 
              (Math.min(...framesPredictions.map(f => f.confidence)) * 100).toFixed(1) + '%'
              : '0%'}
          </div>
          <div className="text-sm text-gray-600">Lowest Confidence</div>
        </div>
      </div>
    </div>
  );
};

export default ConfidenceChart;
