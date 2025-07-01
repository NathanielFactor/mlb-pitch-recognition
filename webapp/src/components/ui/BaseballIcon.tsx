import React from 'react';
import { motion } from 'framer-motion';

interface BaseballIconProps {
  size?: number;
  spinning?: boolean;
  className?: string;
}

const BaseballIcon: React.FC<BaseballIconProps> = ({ 
  size = 24, 
  spinning = false, 
  className = '' 
}) => {
  return (
    <motion.div
      className={`inline-block ${spinning ? 'baseball-spin' : ''} ${className}`}
      animate={spinning ? { rotate: 360 } : {}}
      transition={spinning ? { duration: 2, repeat: Infinity, ease: "linear" } : {}}
    >
      <svg
        width={size}
        height={size}
        viewBox="0 0 24 24"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
      >
        {/* Baseball circle */}
        <circle
          cx="12"
          cy="12"
          r="10"
          fill="white"
          stroke="#e5e7eb"
          strokeWidth="1"
        />
        
        {/* Baseball stitching */}
        <path
          d="M8 6.5C8.5 7.5 9.5 8.5 10.5 9C11.5 9.5 12.5 10.5 13 11.5"
          stroke="#dc2626"
          strokeWidth="1.5"
          strokeLinecap="round"
          fill="none"
        />
        <path
          d="M11 12.5C11.5 13.5 12.5 14.5 13.5 15C14.5 15.5 15.5 16.5 16 17.5"
          stroke="#dc2626"
          strokeWidth="1.5"
          strokeLinecap="round"
          fill="none"
        />
        <path
          d="M16 6.5C15.5 7.5 14.5 8.5 13.5 9C12.5 9.5 11.5 10.5 11 11.5"
          stroke="#dc2626"
          strokeWidth="1.5"
          strokeLinecap="round"
          fill="none"
        />
        <path
          d="M13 12.5C12.5 13.5 11.5 14.5 10.5 15C9.5 15.5 8.5 16.5 8 17.5"
          stroke="#dc2626"
          strokeWidth="1.5"
          strokeLinecap="round"
          fill="none"
        />
      </svg>
    </motion.div>
  );
};

export default BaseballIcon;