import React from 'react';
import { motion } from 'framer-motion';

interface CardProps {
  children: React.ReactNode;
  className?: string;
  variant?: 'base' | 'elevated' | 'floating';
  hover?: boolean;
  glow?: boolean;
}

const Card: React.FC<CardProps> = ({ 
  children, 
  className = '', 
  variant = 'base',
  hover = true, 
  glow = false
}) => {
  const baseClasses = 'backdrop-blur-md border overflow-hidden transition-all duration-300';
  
  const variants = {
    base: 'bg-white/12 rounded-2xl shadow-lg border-white/20',
    elevated: 'bg-white/18 rounded-2xl shadow-xl border-white/25',
    floating: 'bg-white/25 rounded-3xl shadow-2xl border-white/30'
  };
  
  const glowClasses = glow ? 'shadow-white/25 shadow-2xl' : '';
  
  const combinedClassName = `${baseClasses} ${variants[variant]} ${glowClasses} ${className}`;

  const cardVariants = {
    initial: { y: 0, scale: 1 },
    hover: hover ? { 
      y: variant === 'floating' ? -8 : -4, 
      scale: variant === 'floating' ? 1.03 : 1.02,
      boxShadow: variant === 'floating' 
        ? '0 25px 50px -12px rgba(0, 0, 0, 0.25)' 
        : '0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04)'
    } : { y: 0, scale: 1 }
  };

  return (
    <motion.div
      className={combinedClassName}
      variants={cardVariants}
      initial="initial"
      whileHover="hover"
      transition={{ 
        type: "spring",
        stiffness: 400,
        damping: 17
      }}
    >
      {children}
    </motion.div>
  );
};

export default Card;