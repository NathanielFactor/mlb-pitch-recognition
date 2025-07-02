import React from 'react';
import { motion } from 'framer-motion';

interface ProgressBarProps {
  progress: number;
  variant?: 'primary' | 'success' | 'warning' | 'danger';
  size?: 'sm' | 'md' | 'lg';
  showPercentage?: boolean;
  animated?: boolean;
  className?: string;
}

const ProgressBar: React.FC<ProgressBarProps> = ({
  progress,
  variant = 'primary',
  size = 'md',
  showPercentage = true,
  animated = true,
  className = ''
}) => {
  const clampedProgress = Math.min(Math.max(progress, 0), 100);

  const containerClasses = {
    sm: 'h-2',
    md: 'h-3',
    lg: 'h-4'
  };

  const variants = {
    primary: 'bg-gradient-to-r from-blue-500 to-blue-600',
    success: 'bg-gradient-to-r from-green-500 to-green-600',
    warning: 'bg-gradient-to-r from-orange-500 to-orange-600',
    danger: 'bg-gradient-to-r from-red-500 to-red-600'
  };

  return (
    <div className={`w-full ${className}`}>
      {showPercentage && (
        <div className="flex justify-between items-center mb-2">
          <span className="text-sm font-medium text-gray-700">Progress</span>
          <span className="text-sm font-medium text-gray-700">{Math.round(clampedProgress)}%</span>
        </div>
      )}
      <div className={`w-full bg-gray-200 rounded-full overflow-hidden ${containerClasses[size]}`}>
        <motion.div
          className={`h-full ${variants[variant]} ${animated ? 'transition-all duration-300' : ''}`}
          initial={{ width: 0 }}
          animate={{ width: `${clampedProgress}%` }}
          transition={{ duration: animated ? 0.5 : 0, ease: "easeOut" }}
        />
      </div>
    </div>
  );
};

export default ProgressBar;