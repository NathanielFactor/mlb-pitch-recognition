import { PredictionResult, UploadResponse, ApiError, VideoUpload } from '../types';

export interface StatusResponse {
  job_id: string;
  status: 'processing' | 'completed' | 'failed';
  progress: number;
  message: string;
}

export interface ApiConfig {
  baseUrl: string;
  timeout?: number;
}

export class ApiClient {
  private baseUrl: string;
  private timeout: number;

  constructor(config: ApiConfig) {
    this.baseUrl = config.baseUrl;
    this.timeout = config.timeout || 30000; // 30 seconds default
  }

  private async makeRequest<T>(endpoint: string, options: RequestInit = {}): Promise<T> {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.timeout);

    try {
      const response = await fetch(`${this.baseUrl}${endpoint}`, {
        ...options,
        signal: controller.signal,
        headers: {
          'Content-Type': 'application/json',
          ...options.headers,
        },
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        const error = await response.json().catch(() => ({ detail: 'Request failed' }));
        throw new Error(error.detail || `HTTP ${response.status}`);
      }

      return response.json();
    } catch (err) {
      clearTimeout(timeoutId);
      throw err;
    }
  }

  // Web upload method (for webapp)
  async uploadVideo(file: File): Promise<UploadResponse> {
    const formData = new FormData();
    formData.append('file', file);

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.timeout);

    try {
      const response = await fetch(`${this.baseUrl}/api/upload`, {
        method: 'POST',
        body: formData,
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        const error = await response.json().catch(() => ({ detail: 'Upload failed' }));
        throw new Error(error.detail || 'Upload failed');
      }

      return response.json();
    } catch (err) {
      clearTimeout(timeoutId);
      throw err;
    }
  }

  // Mobile upload method (for React Native)
  async uploadVideoMobile(video: VideoUpload): Promise<UploadResponse> {
    const formData = new FormData();
    
    // React Native FormData format
    formData.append('file', {
      uri: video.uri,
      name: video.name,
      type: video.type,
    } as any);

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.timeout);

    try {
      const response = await fetch(`${this.baseUrl}/api/upload`, {
        method: 'POST',
        body: formData,
        signal: controller.signal,
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        const error = await response.json().catch(() => ({ detail: 'Upload failed' }));
        throw new Error(error.detail || 'Upload failed');
      }

      return response.json();
    } catch (err) {
      clearTimeout(timeoutId);
      throw err;
    }
  }

  async getJobStatus(jobId: string): Promise<StatusResponse> {
    return this.makeRequest<StatusResponse>(`/api/status/${jobId}`);
  }

  async getPredictionResult(jobId: string): Promise<PredictionResult> {
    return this.makeRequest<PredictionResult>(`/api/predict/${jobId}`);
  }

  async checkHealth(): Promise<{ status: string; mock_mode: boolean }> {
    return this.makeRequest<{ status: string; mock_mode: boolean }>('/api/health');
  }
}

// Factory functions for different platforms
export const createWebApiClient = (baseUrl?: string): ApiClient => {
  const getApiBaseUrl = () => {
    if (baseUrl) return baseUrl;
    if (process.env.NEXT_PUBLIC_API_URL) return process.env.NEXT_PUBLIC_API_URL;
    
    if (typeof window !== 'undefined' && window.location.hostname.includes('vercel.app')) {
      return `https://${window.location.hostname}`;
    }
    
    return 'http://localhost:8000';
  };

  return new ApiClient({ baseUrl: getApiBaseUrl() });
};

export const createMobileApiClient = (baseUrl: string): ApiClient => {
  return new ApiClient({ 
    baseUrl,
    timeout: 60000, // Longer timeout for mobile
  });
};