import React from 'react';

interface BadgeProps {
  children: React.ReactNode;
  variant?: 'fastball' | 'curveball' | 'slider' | 'changeup' | 'knuckleball' | 'cutter' | 'default';
  size?: 'sm' | 'md' | 'lg';
  className?: string;
}

const Badge: React.FC<BadgeProps> = ({ 
  children, 
  variant = 'default', 
  size = 'md', 
  className = '' 
}) => {
  const baseClasses = 'inline-flex items-center font-medium rounded-full';
  
  const variants = {
    fastball: 'bg-red-100 text-red-800 border border-red-200',
    curveball: 'bg-blue-100 text-blue-800 border border-blue-200',
    slider: 'bg-green-100 text-green-800 border border-green-200',
    changeup: 'bg-orange-100 text-orange-800 border border-orange-200',
    knuckleball: 'bg-indigo-100 text-indigo-800 border border-indigo-200',
    cutter: 'bg-purple-100 text-purple-800 border border-purple-200',
    default: 'bg-gray-100 text-gray-800 border border-gray-200'
  };

  const sizes = {
    sm: 'px-2 py-1 text-xs',
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