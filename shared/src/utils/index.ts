// Utility functions shared across web and mobile

export const formatPitchType = (pitchType: string): string => {
  const pitchTypes: Record<string, string> = {
    'FF': 'Four-Seam Fastball',
    'SL': 'Slider',
    'CH': 'Changeup',
    'CU': 'Curveball',
    'SI': 'Sinker',
    'FC': 'Cutter',
    'FS': 'Splitter',
    'KN': 'Knuckleball',
  };
  
  return pitchTypes[pitchType.toUpperCase()] || pitchType;
};

export const formatConfidence = (confidence: number): string => {
  return `${Math.round(confidence * 100)}%`;
};

export const formatDate = (dateString: string): string => {
  const date = new Date(dateString);
  return date.toLocaleDateString('en-US', {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });
};

export const formatProcessingTime = (timeInSeconds: number): string => {
  if (timeInSeconds < 60) {
    return `${Math.round(timeInSeconds)}s`;
  }
  
  const minutes = Math.floor(timeInSeconds / 60);
  const seconds = Math.round(timeInSeconds % 60);
  return `${minutes}m ${seconds}s`;
};

export const validateVideoFile = (filename: string, fileSize?: number): { valid: boolean; error?: string } => {
  const validExtensions = ['.mp4', '.mov', '.avi', '.mkv'];
  const maxSize = 100 * 1024 * 1024; // 100MB
  
  const extension = filename.toLowerCase().substring(filename.lastIndexOf('.'));
  
  if (!validExtensions.includes(extension)) {
    return {
      valid: false,
      error: `Invalid file type. Supported formats: ${validExtensions.join(', ')}`,
    };
  }
  
  if (fileSize && fileSize > maxSize) {
    return {
      valid: false,
      error: `File size too large. Maximum size: ${maxSize / 1024 / 1024}MB`,
    };
  }
  
  return { valid: true };
};

export const generateJobId = (): string => {
  return `job_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
};

export const sleep = (ms: number): Promise<void> => {
  return new Promise(resolve => setTimeout(resolve, ms));
};