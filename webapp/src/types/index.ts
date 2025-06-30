export interface PredictionResult {
  job_id: string;
  status: 'processing' | 'completed' | 'failed';
  pitch_type?: string;
  confidence?: number;
  processing_time?: number;
  upload_date?: string;
  filename?: string;
}

export interface UploadResponse {
  job_id: string;
  message: string;
}

export interface HistoryItem {
  id: string;
  filename: string;
  upload_date: string;
  pitch_type: string;
  confidence: number;
  processing_time: number;
}

export type UploadStatus = 'idle' | 'uploading' | 'processing' | 'completed' | 'error';

export interface ApiError {
  message: string;
  status: number;
}