import React from 'react';

const Header = () => {
  return (
    <header className="bg-white shadow-lg border-b-4 border-blue-500">
      <div className="container mx-auto px-4 py-6">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <div className="bg-blue-500 p-3 rounded-full">
              <svg className="w-8 h-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
              </svg>
            </div>
            <div>
              <h1 className="text-3xl font-bold text-gray-800">
                Driver Distraction Detection
              </h1>
              <p className="text-gray-600 mt-1">
                AI-powered video analysis for driver safety
              </p>
            </div>
          </div>
          
          <div className="hidden md:flex items-center space-x-6">
            <div className="text-center">
              <div className="text-2xl font-bold text-blue-500">AI</div>
              <div className="text-xs text-gray-500">Powered</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-green-500">✓</div>
              <div className="text-xs text-gray-500">Accurate</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-purple-500">⚡</div>
              <div className="text-xs text-gray-500">Fast</div>
            </div>
          </div>
        </div>
      </div>
    </header>
  );
};

export default Header;
