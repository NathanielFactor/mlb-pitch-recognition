import React from 'react';
import { motion } from 'framer-motion';

interface CardProps {
  children: React.ReactNode;
  className?: string;
  hover?: boolean;
  glow?: boolean;
  gradient?: boolean;
}

const Card: React.FC<CardProps> = ({ 
  children, 
  className = '', 
  hover = true, 
  glow = false,
  gradient = false 
}) => {
  const baseClasses = 'bg-white rounded-xl shadow-lg border border-gray-200 overflow-hidden';
  const glowClasses = glow ? 'stadium-glow' : '';
  const gradientClasses = gradient ? 'bg-gradient-to-br from-white to-blue-50' : '';
  
  const combinedClassName = `${baseClasses} ${gradientClasses} ${glowClasses} ${className}`;

  const cardVariants = {
    initial: { y: 0, scale: 1 },
    hover: hover ? { y: -4, scale: 1.02 } : { y: 0, scale: 1 }
  };

  return (
    <motion.div
      className={combinedClassName}
      variants={cardVariants}
      initial="initial"
      whileHover="hover"
      transition={{ duration: 0.2, ease: "easeOut" }}
    >
      {children}
    </motion.div>
  );
};

export default Card;