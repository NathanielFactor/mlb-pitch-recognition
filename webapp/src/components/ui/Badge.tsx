import React from 'react';

interface BadgeProps {
  children: React.ReactNode;
  variant?: 'fastball' | 'curveball' | 'slider' | 'changeup' | 'knuckleball' | 'cutter' | 'success' | 'warning' | 'error' | 'default';
  size?: 'sm' | 'md' | 'lg';
  className?: string;
}

const Badge: React.FC<BadgeProps> = ({ 
  children, 
  variant = 'default', 
  size = 'md', 
  className = '' 
}) => {
  const baseClasses = 'inline-flex items-center font-medium rounded-full backdrop-blur-md border transition-all duration-200';
  
  const variants = {
    fastball: 'bg-white/15 text-white border-white/25',
    curveball: 'bg-white/15 text-white border-white/25',
    slider: 'bg-white/15 text-white border-white/25',
    changeup: 'bg-white/15 text-white border-white/25',
    knuckleball: 'bg-white/15 text-white border-white/25',
    cutter: 'bg-white/15 text-white border-white/25',
    success: 'bg-white/15 text-white border-white/25',
    warning: 'bg-white/15 text-white border-white/25',
    error: 'bg-white/15 text-white border-white/25',
    default: 'bg-white/15 text-white border-white/25'
  };

  const sizes = {
    sm: 'px-2 py-0.5 text-xs',
    md: 'px-2.5 py-1 text-sm',
    lg: 'px-3 py-1.5 text-base'
  };

  const combinedClassName = `${baseClasses} ${variants[variant]} ${sizes[size]} ${className}`;

  return (
    <span className={combinedClassName}>
      {children}
    </span>
  );
};

export default Badge;