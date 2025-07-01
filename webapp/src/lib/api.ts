const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

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

    const response = await fetch(`${this.baseUrl}/api/upload`, {
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
    const response = await fetch(`${this.baseUrl}/api/status/${jobId}`);
    
    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: 'Failed to get status' }));
      throw new Error(error.detail || 'Failed to get status');
    }

    return response.json();
  }

  async getPredictionResult(jobId: string): Promise<PredictionResult> {
    const response = await fetch(`${this.baseUrl}/api/predict/${jobId}`);
    
    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: 'Failed to get prediction' }));
      throw new Error(error.detail || 'Failed to get prediction');
    }

    return response.json();
  }

  async checkHealth(): Promise<{ status: string; mock_mode: boolean }> {
    const response = await fetch(`${this.baseUrl}/api/health`);
    
    if (!response.ok) {
      throw new Error('Health check failed');
    }

    return response.json();
  }
}

export const apiClient = new ApiClient();