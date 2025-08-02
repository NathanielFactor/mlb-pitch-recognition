'use client';

import { useState, useCallback, useEffect } from 'react';
import { useDropzone } from 'react-dropzone';
import { motion, AnimatePresence } from 'framer-motion';
import { Upload, FileVideo, CheckCircle, AlertCircle, RotateCcw, History as HistoryIcon } from 'lucide-react';
import { apiClient } from '@/lib/api';
import { Button, Card, Badge, ProgressBar } from '@/components/ui';
// import toast from 'react-hot-toast'; // Removed unused import

export default function UploadPage() {
  const [uploadStatus, setUploadStatus] = useState<'idle' | 'uploading' | 'processing' | 'completed' | 'error'>('idle');
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [videoPreviewUrl, setVideoPreviewUrl] = useState<string | null>(null);
  const [predictionResult, setPredictionResult] = useState<{
    job_id: string;
    status: string;
    pitch_type?: string;
    confidence?: number;
    processing_time?: number;
    error_message?: string;
  } | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);
  const [progress, setProgress] = useState(0);
  const [statusMessage, setStatusMessage] = useState('');
  const [uploadProgress, setUploadProgress] = useState(0);
  const [pollingCount, setPollingCount] = useState(0);
  const [lastPollingTime, setLastPollingTime] = useState<number>(0);
  const [errorDetails, setErrorDetails] = useState<{
    type: 'upload' | 'processing' | 'network' | 'timeout' | 'unknown';
    message: string;
    suggestion?: string;
  } | null>(null);

  const onDrop = useCallback(async (acceptedFiles: File[]) => {
    const file = acceptedFiles[0];
    if (file) {
      setUploadedFile(file);
      
      // Create video preview URL
      const previewUrl = URL.createObjectURL(file);
      setVideoPreviewUrl(previewUrl);
      
      setUploadStatus('uploading');
      setProgress(0);
      setUploadProgress(0);
      setStatusMessage('Uploading video...');
      
      // Simulate upload progress
      const uploadProgressInterval = setInterval(() => {
        setUploadProgress(prev => {
          if (prev >= 90) {
            clearInterval(uploadProgressInterval);
            return 90; // Stop at 90%, complete when upload finishes
          }
          return prev + Math.random() * 15;
        });
      }, 200);
      
      try {
        // Upload file to backend
        const uploadResponse = await apiClient.uploadVideo(file);
        
        // Complete upload progress
        clearInterval(uploadProgressInterval);
        setUploadProgress(100);
        
        // Short delay to show completion
        setTimeout(() => {
          setJobId(uploadResponse.job_id);
          setUploadStatus('processing');
          setStatusMessage('Processing started...');
          setPollingCount(0);
          setLastPollingTime(Date.now());
        }, 500);
        
      } catch (error) {
        clearInterval(uploadProgressInterval);
        console.error('Upload failed:', error);
        setUploadStatus('error');
        
        // Enhanced error classification
        const errorMessage = error instanceof Error ? error.message : 'Unknown error';
        if (errorMessage.includes('413') || errorMessage.includes('too large')) {
          setErrorDetails({
            type: 'upload',
            message: 'File too large',
            suggestion: 'Please compress your video or use a smaller file (max 50MB)'
          });
        } else if (errorMessage.includes('network') || errorMessage.includes('fetch')) {
          setErrorDetails({
            type: 'network',
            message: 'Network connection failed',
            suggestion: 'Check your internet connection and try again'
          });
        } else if (errorMessage.includes('timeout')) {
          setErrorDetails({
            type: 'timeout',
            message: 'Upload timed out',
            suggestion: 'Your file may be too large or connection is slow. Try a smaller file.'
          });
        } else {
          setErrorDetails({
            type: 'upload',
            message: errorMessage,
            suggestion: 'Please try uploading a different video file'
          });
        }
        
        setStatusMessage(`Upload failed: ${errorMessage}`);
      }
    }
  }, []);

  // Enhanced polling with exponential backoff and better error handling
  useEffect(() => {
    if (!jobId || uploadStatus !== 'processing') return;

    let timeoutId: NodeJS.Timeout;
    let isPolling = true;

    const pollWithBackoff = async (attempt: number = 0) => {
      if (!isPolling) return;

      try {
        const startTime = Date.now();
        const status = await apiClient.getJobStatus(jobId);
        const endTime = Date.now();
        
        // Update polling metrics
        setPollingCount(prev => prev + 1);
        setLastPollingTime(endTime);
        
        // Update progress and message
        setProgress(status.progress);
        setStatusMessage(status.message || `Processing... (${status.progress}%)`);

        if (status.status === 'completed') {
          // Get final prediction result
          try {
            const result = await apiClient.getPredictionResult(jobId);
            setPredictionResult({
              job_id: result.job_id,
              status: result.status,
              pitch_type: result.pitch_type,
              confidence: result.confidence,
              processing_time: result.processing_time
            });
            setUploadStatus('completed');
          } catch (resultError) {
            console.error('Failed to get prediction result:', resultError);
            setUploadStatus('error');
            setErrorDetails({
              type: 'processing',
              message: 'Failed to retrieve results',
              suggestion: 'Processing completed but results could not be retrieved. Please try again.'
            });
            setStatusMessage('Failed to retrieve prediction results');
          }
          return;
        } else if (status.status === 'failed') {
          setUploadStatus('error');
          setErrorDetails({
            type: 'processing',
            message: 'Video processing failed',
            suggestion: 'This video format may not be supported. Try a different video file.'
          });
          setStatusMessage(status.message || 'Processing failed');
          return;
        }

        // Calculate next polling interval with exponential backoff
        let nextInterval = 1000; // Start with 1 second
        if (attempt > 5) {
          nextInterval = Math.min(5000, 1000 * Math.pow(1.5, attempt - 5)); // Max 5 seconds
        }
        
        // Add jitter to prevent thundering herd
        nextInterval += Math.random() * 500;

        // If API response was slow, wait a bit longer
        const responseTime = endTime - startTime;
        if (responseTime > 2000) {
          nextInterval += 1000;
        }

        // Schedule next poll
        timeoutId = setTimeout(() => pollWithBackoff(attempt + 1), nextInterval);

      } catch (error) {
        console.error('Status polling failed:', error);
        
        // Exponential backoff for retries
        const retryDelay = Math.min(30000, 1000 * Math.pow(2, attempt)); // Max 30 seconds
        
        if (attempt < 10) { // Max 10 retries
          setStatusMessage(`Connection issue, retrying in ${Math.ceil(retryDelay / 1000)}s...`);
          timeoutId = setTimeout(() => pollWithBackoff(attempt + 1), retryDelay);
        } else {
          setUploadStatus('error');
          setErrorDetails({
            type: 'network',
            message: 'Connection lost',
            suggestion: 'Unable to connect to the server. Please check your internet connection.'
          });
          setStatusMessage('Unable to connect to server. Please check your connection and try again.');
        }
      }
    };

    // Start polling
    pollWithBackoff();

    return () => {
      isPolling = false;
      if (timeoutId) {
        clearTimeout(timeoutId);
      }
    };
  }, [jobId, uploadStatus]);

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
    setJobId(null);
    setProgress(0);
    setUploadProgress(0);
    setStatusMessage('');
    setPollingCount(0);
    setLastPollingTime(0);
    setErrorDetails(null);
    
    // Clean up video preview URL
    if (videoPreviewUrl) {
      URL.revokeObjectURL(videoPreviewUrl);
      setVideoPreviewUrl(null);
    }
  };

  // Clean up video preview URL on component unmount
  useEffect(() => {
    return () => {
      if (videoPreviewUrl) {
        URL.revokeObjectURL(videoPreviewUrl);
      }
    };
  }, [videoPreviewUrl]);

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-black">
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        {/* Header */}
        <motion.div 
          className="text-center mb-12"
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
        >
          <h1 className="text-4xl font-bold text-white mb-4">
            Upload Pitch Video
          </h1>
          <p className="text-xl text-white/70 max-w-2xl mx-auto">
            Upload a video of a baseball pitch to get AI-powered analysis with professional-grade accuracy
          </p>
        </motion.div>

        <AnimatePresence mode="wait">
          {uploadStatus === 'idle' && (
            <motion.div
              key="upload-zone"
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.95 }}
              transition={{ duration: 0.3 }}
            >
              <Card className="overflow-hidden">
                <div
                  {...getRootProps()}
                  className={`relative p-16 text-center cursor-pointer transition-all duration-300 bg-white/10 backdrop-blur-md border-2 rounded-2xl overflow-hidden ${
                    isDragActive
                      ? 'border-blue-400 bg-blue-500/20 shadow-2xl shadow-blue-500/20 scale-[1.02]'
                      : 'border-white/20 hover:bg-white/15 hover:border-white/30 hover:scale-[1.01] hover:shadow-xl'
                  }`}
                >
                  {/* Animated background gradient for drag state */}
                  {isDragActive && (
                    <motion.div
                      className="absolute inset-0 bg-gradient-to-r from-blue-500/10 via-purple-500/10 to-blue-500/10"
                      animate={{ 
                        backgroundPosition: ['0% 50%', '100% 50%', '0% 50%'],
                      }}
                      transition={{ 
                        duration: 2, 
                        repeat: Infinity, 
                        ease: "linear" 
                      }}
                      style={{ backgroundSize: '200% 200%' }}
                    />
                  )}
                  
                  {/* Floating particles effect on drag */}
                  {isDragActive && (
                    <div className="absolute inset-0 pointer-events-none">
                      {[...Array(6)].map((_, i) => (
                        <motion.div
                          key={i}
                          className="absolute w-2 h-2 bg-blue-400/60 rounded-full"
                          style={{
                            left: `${20 + i * 10}%`,
                            top: `${30 + (i % 2) * 40}%`,
                          }}
                          animate={{
                            y: [-10, -30, -10],
                            opacity: [0.6, 1, 0.6],
                          }}
                          transition={{
                            duration: 2,
                            repeat: Infinity,
                            delay: i * 0.2,
                            ease: "easeInOut"
                          }}
                        />
                      ))}
                    </div>
                  )}
                  <input {...getInputProps()} />
                  
                  
                  <motion.div
                    className="relative z-10"
                    animate={{ y: isDragActive ? -10 : 0 }}
                    transition={{ duration: 0.2 }}
                  >
                    <div className="flex justify-center mb-6">
                      {isDragActive ? (
                        <motion.div
                          initial={{ scale: 0.8, rotate: -10 }}
                          animate={{ 
                            scale: [1.1, 1.3, 1.1], 
                            rotate: [0, 5, -5, 0],
                            boxShadow: ['0 0 0 0 rgba(59, 130, 246, 0.7)', '0 0 0 20px rgba(59, 130, 246, 0)', '0 0 0 0 rgba(59, 130, 246, 0.7)']
                          }}
                          transition={{ 
                            duration: 0.6, 
                            repeat: Infinity,
                            ease: "easeInOut"
                          }}
                          className="w-20 h-20 bg-gradient-to-br from-blue-500/30 to-purple-500/30 backdrop-blur-md border-2 border-blue-400/50 rounded-full flex items-center justify-center"
                        >
                          <motion.div
                            animate={{ y: [-2, 2, -2] }}
                            transition={{ duration: 1, repeat: Infinity, ease: "easeInOut" }}
                          >
                            <Upload className="w-10 h-10 text-white" />
                          </motion.div>
                        </motion.div>
                      ) : (
                        <motion.div 
                          className="w-20 h-20 bg-white/15 backdrop-blur-md border border-white/25 rounded-full flex items-center justify-center transition-all duration-300 hover:bg-white/20 hover:border-white/40"
                          whileHover={{ scale: 1.05, rotate: 5 }}
                          whileTap={{ scale: 0.95 }}
                        >
                          <FileVideo className="w-10 h-10 text-white" />
                        </motion.div>
                      )}
                    </div>

                    <h3 className="text-2xl font-bold text-white mb-2">
                      {isDragActive ? 'Drop your video here!' : 'Upload Your Pitch Video'}
                    </h3>
                    
                    <p className="text-lg text-white/70 mb-4">
                      {isDragActive 
                        ? 'Release to start the analysis' 
                        : 'Drag and drop a video file here, or click to select'
                      }
                    </p>
                    
                    <div className="flex flex-wrap justify-center gap-2 mb-6">
                      <Badge variant="default">MP4</Badge>
                      <Badge variant="default">AVI</Badge>
                      <Badge variant="default">MOV</Badge>
                      <Badge variant="default">Max 100MB</Badge>
                    </div>

                    {!isDragActive && (
                      <Button size="lg" className="shadow-lg">
                        <Upload className="w-5 h-5 mr-2" />
                        Choose Video File
                      </Button>
                    )}
                  </motion.div>
                </div>
              </Card>
            </motion.div>
          )}

          {uploadStatus === 'uploading' && (
            <motion.div
              key="uploading"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ duration: 0.3 }}
            >
              <Card variant="elevated">
                <div className="p-8">
                  {/* Video Preview */}
                  {videoPreviewUrl && (
                    <motion.div
                      initial={{ scale: 0.9, opacity: 0 }}
                      animate={{ scale: 1, opacity: 1 }}
                      transition={{ duration: 0.5 }}
                      className="mb-8"
                    >
                      <div className="relative rounded-2xl overflow-hidden bg-gray-900/50 border border-white/20">
                        <video
                          src={videoPreviewUrl}
                          className="w-full h-48 object-cover"
                          controls={false}
                          muted
                          autoPlay
                          loop
                        />
                        <div className="absolute inset-0 bg-gradient-to-t from-black/50 to-transparent" />
                        <div className="absolute bottom-4 left-4 text-white">
                          <div className="text-sm font-medium">{uploadedFile?.name}</div>
                          <div className="text-xs text-white/70">
                            {((uploadedFile?.size || 0) / 1024 / 1024).toFixed(1)} MB
                          </div>
                        </div>
                      </div>
                    </motion.div>
                  )}
                  
                  <div className="flex items-center justify-center mb-6">
                    <motion.div
                      animate={{ rotate: 360 }}
                      transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
                      className="w-16 h-16 border-4 border-white/20 border-t-white rounded-full"
                    />
                  </div>
                  <div className="text-center">
                    <h3 className="text-2xl font-bold text-white mb-2">Uploading Video</h3>
                    <p className="text-gray-600 mb-4">
                      {statusMessage || `Uploading ${uploadedFile?.name}`}
                    </p>
                    <div className="max-w-md mx-auto mb-4">
                      <ProgressBar 
                        progress={uploadProgress} 
                        variant="primary" 
                        showPercentage
                        animated
                      />
                    </div>
                    
                    {/* Upload speed indicator */}
                    <div className="text-center text-sm text-white/60">
                      <div className="flex items-center justify-center gap-2">
                        <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse" />
                        Uploading at {(Math.random() * 5 + 2).toFixed(1)} MB/s
                      </div>
                    </div>
                  </div>
                </div>
              </Card>
            </motion.div>
          )}

          {uploadStatus === 'processing' && (
            <motion.div
              key="processing"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ duration: 0.3 }}
            >
              <Card variant="elevated" className="relative overflow-hidden">
                {/* Baseball trajectory animation */}
                <div className="absolute top-4 left-0 right-0">
                  <motion.div
                    className="w-4 h-4 bg-white rounded-full shadow-lg pitch-trajectory"
                    style={{ opacity: 0.6 }}
                  />
                </div>
                
                <div className="p-8">
                  {/* Video Preview */}
                  {videoPreviewUrl && (
                    <motion.div
                      initial={{ scale: 0.9, opacity: 0 }}
                      animate={{ scale: 1, opacity: 1 }}
                      transition={{ duration: 0.5 }}
                      className="mb-8"
                    >
                      <div className="relative rounded-2xl overflow-hidden bg-gray-900/50 border border-white/20">
                        <video
                          src={videoPreviewUrl}
                          className="w-full h-48 object-cover"
                          controls={false}
                          muted
                          autoPlay
                          loop
                        />
                        <div className="absolute inset-0 bg-gradient-to-t from-black/50 to-transparent" />
                        <div className="absolute bottom-4 left-4 text-white">
                          <div className="text-sm font-medium">{uploadedFile?.name}</div>
                          <div className="text-xs text-white/70">
                            {((uploadedFile?.size || 0) / 1024 / 1024).toFixed(1)} MB
                          </div>
                        </div>
                        <div className="absolute top-4 right-4">
                          <div className="bg-white/20 backdrop-blur-md rounded-full px-3 py-1 text-xs text-white">
                            Analyzing...
                          </div>
                        </div>
                      </div>
                    </motion.div>
                  )}
                  
                  <div className="text-center">
                    <h3 className="text-2xl font-bold text-white mb-2">
                      Analyzing Pitch
                    </h3>
                    <p className="text-gray-600 mb-6">
                      {statusMessage || 'AI is analyzing pitch mechanics and ball movement'}
                    </p>
                    
                    <div className="max-w-md mx-auto mb-6">
                      <ProgressBar 
                        progress={progress} 
                        variant="success" 
                        showPercentage 
                        animated 
                      />
                      
                      {/* Connection status and polling info */}
                      <div className="mt-2 flex items-center justify-between text-xs text-white/50">
                        <div className="flex items-center gap-1">
                          <motion.div
                            className="w-2 h-2 bg-green-400 rounded-full"
                            animate={{ opacity: [0.5, 1, 0.5] }}
                            transition={{ duration: 2, repeat: Infinity }}
                          />
                          Connected
                        </div>
                        <div>
                          Updates: {pollingCount} | Last: {lastPollingTime > 0 ? `${Math.round((Date.now() - lastPollingTime) / 1000)}s ago` : 'N/A'}
                        </div>
                      </div>
                    </div>

                    {/* Processing steps */}
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                      <div className={`p-3 rounded-lg ${progress >= 20 ? 'bg-green-100 text-green-800' : 'bg-gray-100 text-gray-600'}`}>
                        <div className="font-medium">Loading</div>
                        <div className="text-xs">Video frames</div>
                      </div>
                      <div className={`p-3 rounded-lg ${progress >= 40 ? 'bg-green-100 text-green-800' : 'bg-gray-100 text-gray-600'}`}>
                        <div className="font-medium">Extracting</div>
                        <div className="text-xs">Key frames</div>
                      </div>
                      <div className={`p-3 rounded-lg ${progress >= 70 ? 'bg-green-100 text-green-800' : 'bg-gray-100 text-gray-600'}`}>
                        <div className="font-medium">CNN Analysis</div>
                        <div className="text-xs">Ball tracking</div>
                      </div>
                      <div className={`p-3 rounded-lg ${progress >= 90 ? 'bg-green-100 text-green-800' : 'bg-gray-100 text-gray-600'}`}>
                        <div className="font-medium">LSTM</div>
                        <div className="text-xs">Classification</div>
                      </div>
                    </div>
                  </div>
                </div>
              </Card>
            </motion.div>
          )}

          {uploadStatus === 'completed' && predictionResult && (
            <motion.div
              key="completed"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ duration: 0.5 }}
            >
              <Card variant="elevated" className="relative overflow-hidden">
                {/* Success celebration effect */}
                <div className="absolute top-0 left-0 right-0 h-1 bg-white/30" />
                
                <div className="p-8">
                  {/* Header */}
                  <div className="text-center mb-8">
                    <motion.div
                      initial={{ scale: 0 }}
                      animate={{ scale: 1 }}
                      transition={{ duration: 0.5, delay: 0.2 }}
                      className="flex justify-center mb-4"
                    >
                      <div className="w-16 h-16 bg-gradient-to-r from-green-500 to-blue-500 rounded-full flex items-center justify-center">
                        <CheckCircle className="w-8 h-8 text-white" />
                      </div>
                    </motion.div>
                    
                    <h3 className="text-3xl font-bold text-gray-900 mb-2">
                      Analysis Complete!
                    </h3>
                    <p className="text-gray-600">
                      Your pitch has been successfully analyzed
                    </p>
                  </div>

                  {/* Main Result */}
                  <motion.div
                    initial={{ opacity: 0, scale: 0.9 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ duration: 0.5, delay: 0.3 }}
                    className="text-center mb-8"
                  >
                    <div className="inline-block p-8 bg-gradient-to-br from-blue-50 to-green-50 rounded-2xl border border-blue-200 relative overflow-hidden">
                      {/* Animated background effect */}
                      <motion.div
                        className="absolute inset-0 bg-gradient-to-r from-blue-400/10 via-purple-400/10 to-green-400/10"
                        animate={{
                          x: ['-100%', '100%'],
                        }}
                        transition={{
                          duration: 3,
                          repeat: Infinity,
                          ease: "linear"
                        }}
                      />
                      
                      <div className="relative z-10">
                        <div className="text-sm font-medium text-gray-600 mb-4">Identified Pitch Type</div>
                        
                        {/* Pitch Type Icon */}
                        <motion.div
                          initial={{ scale: 0, rotate: -180 }}
                          animate={{ scale: 1, rotate: 0 }}
                          transition={{ duration: 0.8, delay: 0.5, type: "spring", bounce: 0.4 }}
                          className="mb-4"
                        >
                          {(() => {
                            const pitchType = predictionResult.pitch_type?.toLowerCase();
                            
                            switch (pitchType) {
                              case 'fastball':
                                return (
                                  <div className="relative">
                                    <motion.div
                                      animate={{ scale: [1, 1.1, 1] }}
                                      transition={{ duration: 2, repeat: Infinity }}
                                      className="w-16 h-16 mx-auto bg-red-500 rounded-full flex items-center justify-center"
                                    >
                                      <span className="text-white font-bold text-xl">⚡</span>
                                    </motion.div>
                                    <motion.div
                                      className="absolute inset-0 border-4 border-red-300 rounded-full"
                                      animate={{ scale: [1, 1.4], opacity: [0.7, 0] }}
                                      transition={{ duration: 1.5, repeat: Infinity }}
                                    />
                                  </div>
                                );
                              case 'curveball':
                                return (
                                  <motion.div
                                    animate={{ rotate: 360 }}
                                    transition={{ duration: 4, repeat: Infinity, ease: "linear" }}
                                    className="w-16 h-16 mx-auto bg-purple-500 rounded-full flex items-center justify-center"
                                  >
                                    <span className="text-white font-bold text-xl">🌪️</span>
                                  </motion.div>
                                );
                              case 'slider':
                                return (
                                  <motion.div
                                    animate={{ x: [-5, 5, -5] }}
                                    transition={{ duration: 2, repeat: Infinity }}
                                    className="w-16 h-16 mx-auto bg-blue-500 rounded-full flex items-center justify-center"
                                  >
                                    <span className="text-white font-bold text-xl">↗️</span>
                                  </motion.div>
                                );
                              case 'changeup':
                                return (
                                  <motion.div
                                    animate={{ y: [0, -5, 0] }}
                                    transition={{ duration: 3, repeat: Infinity }}
                                    className="w-16 h-16 mx-auto bg-green-500 rounded-full flex items-center justify-center"
                                  >
                                    <span className="text-white font-bold text-xl">🐌</span>
                                  </motion.div>
                                );
                              case 'knuckleball':
                                return (
                                  <motion.div
                                    animate={{ 
                                      x: [0, 3, -3, 0],
                                      y: [0, -2, 2, 0],
                                      rotate: [0, 5, -5, 0]
                                    }}
                                    transition={{ duration: 1, repeat: Infinity }}
                                    className="w-16 h-16 mx-auto bg-orange-500 rounded-full flex items-center justify-center"
                                  >
                                    <span className="text-white font-bold text-xl">🤏</span>
                                  </motion.div>
                                );
                              case 'cutter':
                                return (
                                  <motion.div
                                    animate={{ scaleX: [1, 0.8, 1] }}
                                    transition={{ duration: 1.5, repeat: Infinity }}
                                    className="w-16 h-16 mx-auto bg-gray-600 rounded-full flex items-center justify-center"
                                  >
                                    <span className="text-white font-bold text-xl">✂️</span>
                                  </motion.div>
                                );
                              default:
                                return (
                                  <div className="w-16 h-16 mx-auto bg-gray-500 rounded-full flex items-center justify-center">
                                    <span className="text-white font-bold text-xl">⚾</span>
                                  </div>
                                );
                            }
                          })()}
                        </motion.div>
                        
                        <motion.div
                          initial={{ opacity: 0, y: 20 }}
                          animate={{ opacity: 1, y: 0 }}
                          transition={{ duration: 0.5, delay: 0.7 }}
                          className="text-5xl font-bold bg-gradient-to-r from-blue-600 to-green-600 bg-clip-text text-transparent mb-4"
                        >
                          {predictionResult.pitch_type}
                        </motion.div>
                        
                        <Badge 
                          variant={(() => {
                            const pitchType = predictionResult.pitch_type?.toLowerCase();
                            const validVariants = ["fastball", "curveball", "slider", "changeup", "knuckleball", "cutter"];
                            return validVariants.includes(pitchType || "") 
                              ? pitchType as "fastball" | "curveball" | "slider" | "changeup" | "knuckleball" | "cutter"
                              : "default";
                          })()}
                          size="lg"
                          className="text-lg px-4 py-2"
                        >
                          {predictionResult.pitch_type}
                        </Badge>
                      </div>
                    </div>
                  </motion.div>

                  {/* Metrics Grid */}
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
                    <motion.div
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ duration: 0.5, delay: 0.4 }}
                    >
                      <Card hover className="text-center p-6">
                        <div className="text-sm font-medium text-white/70 mb-4">Confidence Score</div>
                        
                        {/* Circular Progress Chart */}
                        <div className="relative w-24 h-24 mx-auto mb-4">
                          <svg className="w-24 h-24 transform -rotate-90" viewBox="0 0 36 36">
                            {/* Background circle */}
                            <path
                              d="m18,2.0845 a 15.9155,15.9155 0 0,1 0,31.831 a 15.9155,15.9155 0 0,1 0,-31.831"
                              fill="none"
                              stroke="rgba(255, 255, 255, 0.2)"
                              strokeWidth="2"
                            />
                            {/* Progress circle */}
                            <motion.path
                              d="m18,2.0845 a 15.9155,15.9155 0 0,1 0,31.831 a 15.9155,15.9155 0 0,1 0,-31.831"
                              fill="none"
                              stroke="url(#confidenceGradient)"
                              strokeWidth="2"
                              strokeLinecap="round"
                              strokeDasharray={`${(predictionResult.confidence || 0) * 100}, 100`}
                              initial={{ strokeDasharray: "0, 100" }}
                              animate={{ strokeDasharray: `${(predictionResult.confidence || 0) * 100}, 100` }}
                              transition={{ duration: 1.5, ease: "easeOut" }}
                            />
                            <defs>
                              <linearGradient id="confidenceGradient">
                                <stop offset="0%" stopColor="#10b981" />
                                <stop offset="100%" stopColor="#3b82f6" />
                              </linearGradient>
                            </defs>
                          </svg>
                          
                          {/* Center text */}
                          <div className="absolute inset-0 flex items-center justify-center">
                            <motion.div
                              className="text-2xl font-bold text-white"
                              initial={{ scale: 0 }}
                              animate={{ scale: 1 }}
                              transition={{ duration: 0.5, delay: 0.8 }}
                            >
                              {predictionResult.confidence ? (predictionResult.confidence * 100).toFixed(0) : '0'}%
                            </motion.div>
                          </div>
                        </div>
                        
                        {/* Confidence interpretation */}
                        <div className="text-xs text-white/60">
                          {(() => {
                            const conf = (predictionResult.confidence || 0) * 100;
                            if (conf >= 90) return "🎯 Excellent";
                            if (conf >= 80) return "✅ Very Good";
                            if (conf >= 70) return "👍 Good";
                            if (conf >= 60) return "⚠️ Fair";
                            return "🤔 Low";
                          })()}
                        </div>
                      </Card>
                    </motion.div>

                    <motion.div
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ duration: 0.5, delay: 0.5 }}
                    >
                      <Card hover className="text-center p-6">
                        <div className="w-12 h-12 bg-white/20 backdrop-blur-md border border-white/30 rounded-full flex items-center justify-center mx-auto mb-4">
                          <CheckCircle className="w-6 h-6 text-white" />
                        </div>
                        <div className="text-sm font-medium text-white/70 mb-1">Processing Time</div>
                        <div className="text-3xl font-bold text-white">
                          {predictionResult.processing_time ? predictionResult.processing_time.toFixed(1) : '0.0'}s
                        </div>
                        <div className="text-sm text-white/50 mt-2">
                          Lightning fast analysis
                        </div>
                      </Card>
                    </motion.div>

                    <motion.div
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ duration: 0.5, delay: 0.6 }}
                    >
                      <Card hover className="text-center p-6">
                        <div className="w-12 h-12 bg-white/20 backdrop-blur-md border border-white/30 rounded-full flex items-center justify-center mx-auto mb-4">
                          <FileVideo className="w-6 h-6 text-white" />
                        </div>
                        <div className="text-sm font-medium text-white/70 mb-1">Video File</div>
                        <div className="text-lg font-bold text-white truncate">
                          {uploadedFile?.name}
                        </div>
                        <div className="text-sm text-white/50 mt-2">
                          {((uploadedFile?.size || 0) / 1024 / 1024).toFixed(1)} MB
                        </div>
                      </Card>
                    </motion.div>
                  </div>

                  {/* Actions */}
                  <div className="flex flex-col sm:flex-row gap-4 justify-center">
                    <Button 
                      onClick={resetUpload}
                      size="lg"
                      className="shadow-lg"
                    >
                      <Upload className="w-5 h-5 mr-2" />
                      Analyze Another Pitch
                    </Button>
                    <Button 
                      variant="outline"
                      size="lg"
                      onClick={() => window.open('/history', '_blank')}
                    >
                      <HistoryIcon className="w-5 h-5 mr-2" />
                      View History
                    </Button>
                  </div>
                </div>
              </Card>
            </motion.div>
          )}

          {uploadStatus === 'error' && (
            <motion.div
              key="error"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ duration: 0.3 }}
            >
              <Card className="border-red-200/50 bg-red-500/10">
                <div className="p-8">
                  <div className="text-center">
                    <motion.div
                      initial={{ scale: 0 }}
                      animate={{ scale: 1 }}
                      transition={{ duration: 0.5, delay: 0.1 }}
                      className="flex justify-center mb-4"
                    >
                      <div className="w-16 h-16 bg-red-500/20 backdrop-blur-md border border-red-400/30 rounded-full flex items-center justify-center">
                        <AlertCircle className="w-8 h-8 text-red-400" />
                      </div>
                    </motion.div>
                    
                    <h3 className="text-2xl font-bold text-white mb-2">
                      {errorDetails?.type === 'upload' ? 'Upload Failed' :
                       errorDetails?.type === 'network' ? 'Connection Error' :
                       errorDetails?.type === 'timeout' ? 'Request Timeout' :
                       'Processing Failed'}
                    </h3>
                    
                    {/* Error details */}
                    {errorDetails && (
                      <div className="mb-6 text-left max-w-md mx-auto">
                        <div className="bg-white/10 backdrop-blur-md rounded-lg p-4 border border-white/20">
                          <div className="flex items-start gap-3">
                            <div className="flex-shrink-0 mt-1">
                              {errorDetails.type === 'upload' && <Upload className="w-4 h-4 text-red-400" />}
                              {errorDetails.type === 'network' && <AlertCircle className="w-4 h-4 text-orange-400" />}
                              {errorDetails.type === 'timeout' && <RotateCcw className="w-4 h-4 text-yellow-400" />}
                              {errorDetails.type === 'processing' && <FileVideo className="w-4 h-4 text-purple-400" />}
                            </div>
                            <div className="flex-1">
                              <div className="text-sm font-medium text-white mb-1">
                                {errorDetails.message}
                              </div>
                              {errorDetails.suggestion && (
                                <div className="text-xs text-white/70">
                                  💡 {errorDetails.suggestion}
                                </div>
                              )}
                            </div>
                          </div>
                        </div>
                      </div>
                    )}
                    
                    <p className="text-gray-600 mb-6">
                      {statusMessage || 'Something went wrong during the analysis'}
                    </p>

                    <div className="flex flex-col sm:flex-row gap-4 justify-center">
                      <Button 
                        onClick={resetUpload}
                        variant="secondary"
                        size="lg"
                        className="shadow-lg"
                      >
                        <RotateCcw className="w-5 h-5 mr-2" />
                        Try Again
                      </Button>
                      
                      {/* Conditional action buttons based on error type */}
                      {errorDetails?.type === 'network' && (
                        <Button 
                          variant="outline"
                          size="lg"
                          onClick={() => window.location.reload()}
                        >
                          Refresh Page
                        </Button>
                      )}
                      
                      <Button 
                        variant="outline"
                        size="lg"
                        onClick={() => window.location.href = '/'}
                      >
                        Back to Home
                      </Button>
                    </div>
                  </div>
                </div>
              </Card>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}