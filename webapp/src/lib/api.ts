// Dynamic API URL configuration for different environments
const getApiBaseUrl = () => {
  // Production: Use Render backend URL (set via environment variable)
  if (process.env.NEXT_PUBLIC_RENDER_API_URL) {
    return process.env.NEXT_PUBLIC_RENDER_API_URL;
  }
  
  // Fallback: Legacy API URL environment variable
  if (process.env.NEXT_PUBLIC_API_URL) {
    return process.env.NEXT_PUBLIC_API_URL;
  }
  
  // Production fallback: Use Render API URL when deployed on Vercel
  if (typeof window !== 'undefined' && window.location.hostname.includes('vercel.app')) {
    return 'https://mlb-pitch-api-albt.onrender.com';
  }
  
  // Development: Use localhost backend
  return 'http://localhost:8000';
};

const API_BASE_URL = getApiBaseUrl();

export interface UploadResponse {
  job_id: string;
  message: string;
  filename: string;
}

export interface StatusResponse {
  job_id: string;
  status: 'processing' | 'completed' | 'failed';
  progress: number;
  message: string;
}

export interface PredictionResult {
  job_id: string;
  status: 'processing' | 'completed' | 'failed';
  pitch_type?: string;
  confidence?: number;
  processing_time?: number;
  error_message?: string;
}

class ApiClient {
  private baseUrl: string;

  constructor(baseUrl: string = API_BASE_URL) {
    this.baseUrl = baseUrl;
  }

  async uploadVideo(file: File): Promise<UploadResponse> {
    const formData = new FormData();
    formData.append('file', file);

    // Ensure no double slashes in URL
    const url = `${this.baseUrl.replace(/\/$/, '')}/api/upload`;
    console.log('Making API call to:', url); // Debug logging
    
    const response = await fetch(url, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: 'Upload failed' }));
      throw new Error(error.detail || 'Upload failed');
    }

    return response.json();
  }

  async getJobStatus(jobId: string): Promise<StatusResponse> {
    const url = `${this.baseUrl.replace(/\/$/, '')}/api/status/${jobId}`;
    const response = await fetch(url);
    
    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: 'Failed to get status' }));
      throw new Error(error.detail || 'Failed to get status');
    }

    return response.json();
  }

  async getPredictionResult(jobId: string): Promise<PredictionResult> {
    const url = `${this.baseUrl.replace(/\/$/, '')}/api/predict/${jobId}`;
    const response = await fetch(url);
    
    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: 'Failed to get prediction' }));
      throw new Error(error.detail || 'Failed to get prediction');
    }

    return response.json();
  }

  async checkHealth(): Promise<{ status: string; mock_mode: boolean }> {
    const url = `${this.baseUrl.replace(/\/$/, '')}/api/health`;
    const response = await fetch(url);
    
    if (!response.ok) {
      throw new Error('Health check failed');
    }

    return response.json();
  }
}

export const apiClient = new ApiClient();