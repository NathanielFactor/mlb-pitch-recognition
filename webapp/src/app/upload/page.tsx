'use client';

import { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';

export default function UploadPage() {
  const [uploadStatus, setUploadStatus] = useState<'idle' | 'uploading' | 'processing' | 'completed' | 'error'>('idle');
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [predictionResult, setPredictionResult] = useState<any>(null);

  const onDrop = useCallback((acceptedFiles: File[]) => {
    const file = acceptedFiles[0];
    if (file) {
      setUploadedFile(file);
      setUploadStatus('uploading');
      
      // Simulate upload and processing
      setTimeout(() => {
        setUploadStatus('processing');
        
        // Simulate ML processing
        setTimeout(() => {
          setPredictionResult({
            pitch_type: 'Fastball',
            confidence: 0.89,
            processing_time: 2.3
          });
          setUploadStatus('completed');
        }, 3000);
      }, 1000);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'video/*': ['.mp4', '.avi', '.mov']
    },
    maxFiles: 1,
    disabled: uploadStatus !== 'idle'
  });

  const resetUpload = () => {
    setUploadStatus('idle');
    setUploadedFile(null);
    setPredictionResult(null);
  };

  return (
    <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
      <div className="text-center mb-8">
        <h1 className="text-3xl font-bold text-gray-900">Upload Pitch Video</h1>
        <p className="mt-2 text-gray-600">
          Upload a video of a baseball pitch to get AI-powered pitch type classification
        </p>
      </div>

      {uploadStatus === 'idle' && (
        <div
          {...getRootProps()}
          className={`border-2 border-dashed rounded-lg p-12 text-center transition-colors ${
            isDragActive
              ? 'border-blue-400 bg-blue-50'
              : 'border-gray-300 hover:border-gray-400'
          }`}
        >
          <input {...getInputProps()} />
          <svg className="mx-auto h-12 w-12 text-gray-400" stroke="currentColor" fill="none" viewBox="0 0 48 48">
            <path d="M28 8H12a4 4 0 00-4 4v20m32-12v8m0 0v8a4 4 0 01-4 4H12a4 4 0 01-4-4v-4m32-4l-3.172-3.172a4 4 0 00-5.656 0L28 28M8 32l9.172-9.172a4 4 0 015.656 0L28 28m0 0l4 4m4-24h8m-4-4v8m-12 4h.02" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round" />
          </svg>
          <div className="mt-4">
            <p className="text-lg text-gray-600">
              {isDragActive ? 'Drop the video here' : 'Drag and drop a video file here, or click to select'}
            </p>
            <p className="text-sm text-gray-500 mt-1">
              Supports MP4, AVI, MOV files (max 100MB)
            </p>
          </div>
        </div>
      )}

      {uploadStatus === 'uploading' && (
        <div className="bg-white rounded-lg shadow p-6">
          <div className="flex items-center">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
            <div className="ml-4">
              <h3 className="text-lg font-medium text-gray-900">Uploading...</h3>
              <p className="text-gray-600">Uploading {uploadedFile?.name}</p>
            </div>
          </div>
        </div>
      )}

      {uploadStatus === 'processing' && (
        <div className="bg-white rounded-lg shadow p-6">
          <div className="flex items-center">
            <div className="animate-pulse rounded-full h-8 w-8 bg-blue-600"></div>
            <div className="ml-4">
              <h3 className="text-lg font-medium text-gray-900">Processing Video...</h3>
              <p className="text-gray-600">Analyzing pitch mechanics with AI</p>
            </div>
          </div>
          <div className="mt-4">
            <div className="bg-gray-200 rounded-full h-2">
              <div className="bg-blue-600 h-2 rounded-full animate-pulse" style={{ width: '70%' }}></div>
            </div>
          </div>
        </div>
      )}

      {uploadStatus === 'completed' && predictionResult && (
        <div className="bg-white rounded-lg shadow p-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-medium text-gray-900">Prediction Results</h3>
            <button
              onClick={resetUpload}
              className="text-blue-600 hover:text-blue-800 text-sm font-medium"
            >
              Upload Another
            </button>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="bg-blue-50 rounded-lg p-4">
              <div className="text-sm font-medium text-blue-800">Pitch Type</div>
              <div className="text-2xl font-bold text-blue-900">{predictionResult.pitch_type}</div>
            </div>
            <div className="bg-green-50 rounded-lg p-4">
              <div className="text-sm font-medium text-green-800">Confidence</div>
              <div className="text-2xl font-bold text-green-900">{(predictionResult.confidence * 100).toFixed(1)}%</div>
            </div>
            <div className="bg-gray-50 rounded-lg p-4">
              <div className="text-sm font-medium text-gray-800">Processing Time</div>
              <div className="text-2xl font-bold text-gray-900">{predictionResult.processing_time}s</div>
            </div>
          </div>

          <div className="mt-6">
            <h4 className="text-md font-medium text-gray-900 mb-2">Uploaded Video</h4>
            <div className="flex items-center space-x-2 text-sm text-gray-600">
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
              </svg>
              <span>{uploadedFile?.name}</span>
              <span>({(uploadedFile?.size || 0 / 1024 / 1024).toFixed(1)} MB)</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}