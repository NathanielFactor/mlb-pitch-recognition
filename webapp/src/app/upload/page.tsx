'use client';

import { useState, useCallback, useEffect } from 'react';
import { useDropzone } from 'react-dropzone';
import { motion, AnimatePresence } from 'framer-motion';
import { Upload, FileVideo, CheckCircle, AlertCircle, RotateCcw, History as HistoryIcon } from 'lucide-react';
import { apiClient } from '@/lib/api';
import { Button, Card, Badge, ProgressBar, BaseballIcon } from '@/components/ui';
// import toast from 'react-hot-toast'; // Removed unused import

export default function UploadPage() {
  const [uploadStatus, setUploadStatus] = useState<'idle' | 'uploading' | 'processing' | 'completed' | 'error'>('idle');
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
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

  const onDrop = useCallback(async (acceptedFiles: File[]) => {
    const file = acceptedFiles[0];
    if (file) {
      setUploadedFile(file);
      setUploadStatus('uploading');
      setProgress(0);
      setStatusMessage('Uploading video...');
      
      try {
        // Upload file to backend
        const uploadResponse = await apiClient.uploadVideo(file);
        setJobId(uploadResponse.job_id);
        setUploadStatus('processing');
        setStatusMessage('Processing started...');
        
      } catch (error) {
        console.error('Upload failed:', error);
        setUploadStatus('error');
        setStatusMessage(`Upload failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
      }
    }
  }, []);

  // Poll for job status updates
  useEffect(() => {
    if (!jobId || uploadStatus !== 'processing') return;

    const pollInterval = setInterval(async () => {
      try {
        const status = await apiClient.getJobStatus(jobId);
        setProgress(status.progress);
        setStatusMessage(status.message);

        if (status.status === 'completed') {
          // Get final prediction result
          const result = await apiClient.getPredictionResult(jobId);
          setPredictionResult({
            pitch_type: result.pitch_type,
            confidence: result.confidence,
            processing_time: result.processing_time
          });
          setUploadStatus('completed');
          clearInterval(pollInterval);
        } else if (status.status === 'failed') {
          setUploadStatus('error');
          setStatusMessage('Processing failed');
          clearInterval(pollInterval);
        }
      } catch (error) {
        console.error('Status polling failed:', error);
        setUploadStatus('error');
        setStatusMessage('Failed to get processing status');
        clearInterval(pollInterval);
      }
    }, 1000); // Poll every second

    return () => clearInterval(pollInterval);
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
    setStatusMessage('');
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 via-blue-50 to-green-50">
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        {/* Header */}
        <motion.div 
          className="text-center mb-12"
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
        >
          <div className="flex justify-center mb-4">
            <BaseballIcon size={48} spinning={uploadStatus === 'processing'} />
          </div>
          <h1 className="text-4xl font-bold bg-gradient-to-r from-blue-900 to-green-700 bg-clip-text text-transparent mb-4">
            Upload Pitch Video
          </h1>
          <p className="text-xl text-gray-600 max-w-2xl mx-auto">
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
                  className={`relative p-16 text-center cursor-pointer transition-all duration-300 ${
                    isDragActive
                      ? 'bg-gradient-to-br from-blue-50 to-green-50 border-2 border-blue-400 border-dashed'
                      : 'bg-gradient-to-br from-gray-50 to-blue-50 hover:from-blue-50 hover:to-green-50 border-2 border-gray-300 border-dashed hover:border-blue-400'
                  }`}
                >
                  <input {...getInputProps()} />
                  
                  {/* Baseball field background effect */}
                  <div className="absolute inset-0 opacity-5">
                    <div className="field-gradient h-full w-full" />
                  </div>
                  
                  <motion.div
                    className="relative z-10"
                    animate={{ y: isDragActive ? -10 : 0 }}
                    transition={{ duration: 0.2 }}
                  >
                    <div className="flex justify-center mb-6">
                      {isDragActive ? (
                        <motion.div
                          initial={{ scale: 0.8 }}
                          animate={{ scale: 1.2 }}
                          className="w-20 h-20 bg-gradient-to-r from-blue-500 to-green-500 rounded-full flex items-center justify-center"
                        >
                          <Upload className="w-10 h-10 text-white" />
                        </motion.div>
                      ) : (
                        <div className="w-20 h-20 bg-gradient-to-r from-gray-400 to-gray-500 rounded-full flex items-center justify-center">
                          <FileVideo className="w-10 h-10 text-white" />
                        </div>
                      )}
                    </div>

                    <h3 className="text-2xl font-bold text-gray-900 mb-2">
                      {isDragActive ? 'Drop your video here!' : 'Upload Your Pitch Video'}
                    </h3>
                    
                    <p className="text-lg text-gray-600 mb-4">
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
              <Card gradient glow>
                <div className="p-8">
                  <div className="flex items-center justify-center mb-6">
                    <motion.div
                      animate={{ rotate: 360 }}
                      transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
                      className="w-16 h-16 border-4 border-blue-200 border-t-blue-600 rounded-full"
                    />
                  </div>
                  <div className="text-center">
                    <h3 className="text-2xl font-bold text-gray-900 mb-2">Uploading Video</h3>
                    <p className="text-gray-600 mb-4">
                      {statusMessage || `Uploading ${uploadedFile?.name}`}
                    </p>
                    <div className="max-w-md mx-auto">
                      <ProgressBar progress={25} variant="primary" />
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
              <Card gradient glow className="relative overflow-hidden">
                {/* Baseball trajectory animation */}
                <div className="absolute top-4 left-0 right-0">
                  <motion.div
                    className="w-4 h-4 bg-white rounded-full shadow-lg pitch-trajectory"
                    style={{ opacity: 0.6 }}
                  />
                </div>
                
                <div className="p-8">
                  <div className="flex items-center justify-center mb-6">
                    <BaseballIcon size={64} spinning />
                  </div>
                  
                  <div className="text-center">
                    <h3 className="text-2xl font-bold text-gray-900 mb-2">
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
              <Card gradient className="relative overflow-hidden">
                {/* Success celebration effect */}
                <div className="absolute top-0 left-0 right-0 h-1 bg-gradient-to-r from-green-400 via-blue-500 to-green-400" />
                
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
                    <div className="inline-block p-8 bg-gradient-to-br from-blue-50 to-green-50 rounded-2xl border border-blue-200">
                      <div className="text-sm font-medium text-gray-600 mb-2">Identified Pitch Type</div>
                      <div className="text-5xl font-bold bg-gradient-to-r from-blue-600 to-green-600 bg-clip-text text-transparent mb-4">
                        {predictionResult.pitch_type}
                      </div>
                      <Badge 
                        variant={predictionResult.pitch_type?.toLowerCase() as "default" | "secondary" | "destructive" | "outline"}
                        size="lg"
                        className="text-lg px-4 py-2"
                      >
                        {predictionResult.pitch_type}
                      </Badge>
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
                        <div className="w-12 h-12 bg-gradient-to-r from-green-500 to-green-600 rounded-full flex items-center justify-center mx-auto mb-4">
                          <CheckCircle className="w-6 h-6 text-white" />
                        </div>
                        <div className="text-sm font-medium text-gray-600 mb-1">Confidence Score</div>
                        <div className="text-3xl font-bold text-green-600">
                          {(predictionResult.confidence * 100).toFixed(1)}%
                        </div>
                        <div className="mt-3">
                          <ProgressBar 
                            progress={predictionResult.confidence * 100} 
                            variant="success" 
                            size="sm"
                            showPercentage={false}
                          />
                        </div>
                      </Card>
                    </motion.div>

                    <motion.div
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ duration: 0.5, delay: 0.5 }}
                    >
                      <Card hover className="text-center p-6">
                        <div className="w-12 h-12 bg-gradient-to-r from-blue-500 to-blue-600 rounded-full flex items-center justify-center mx-auto mb-4">
                          <BaseballIcon size={24} />
                        </div>
                        <div className="text-sm font-medium text-gray-600 mb-1">Processing Time</div>
                        <div className="text-3xl font-bold text-blue-600">
                          {predictionResult.processing_time}s
                        </div>
                        <div className="text-sm text-gray-500 mt-2">
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
                        <div className="w-12 h-12 bg-gradient-to-r from-purple-500 to-purple-600 rounded-full flex items-center justify-center mx-auto mb-4">
                          <FileVideo className="w-6 h-6 text-white" />
                        </div>
                        <div className="text-sm font-medium text-gray-600 mb-1">Video File</div>
                        <div className="text-lg font-bold text-purple-600 truncate">
                          {uploadedFile?.name}
                        </div>
                        <div className="text-sm text-gray-500 mt-2">
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
              <Card className="border-red-200">
                <div className="p-8">
                  <div className="text-center">
                    <motion.div
                      initial={{ scale: 0 }}
                      animate={{ scale: 1 }}
                      transition={{ duration: 0.5, delay: 0.1 }}
                      className="flex justify-center mb-4"
                    >
                      <div className="w-16 h-16 bg-gradient-to-r from-red-500 to-red-600 rounded-full flex items-center justify-center">
                        <AlertCircle className="w-8 h-8 text-white" />
                      </div>
                    </motion.div>
                    
                    <h3 className="text-2xl font-bold text-gray-900 mb-2">
                      Processing Failed
                    </h3>
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