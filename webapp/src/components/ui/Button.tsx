import React from 'react';
import { motion, HTMLMotionProps } from 'framer-motion';
import { Loader2 } from 'lucide-react';

interface ButtonProps extends Omit<HTMLMotionProps<"button">, "children"> {
  variant?: 'primary' | 'secondary' | 'floating' | 'outline' | 'ghost';
  size?: 'sm' | 'md' | 'lg';
  loading?: boolean;
  children: React.ReactNode;
}

const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ variant = 'primary', size = 'md', loading = false, children, className = '', disabled, ...props }, ref) => {
    const baseClasses = 'inline-flex items-center justify-center font-medium transition-all duration-300 focus:outline-none focus:ring-2 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed backdrop-blur-md border';
    
    const variants = {
      primary: 'bg-white/15 hover:bg-white/25 text-white border-white/20 hover:border-white/30 shadow-lg hover:shadow-xl focus:ring-white/30 rounded-2xl',
      secondary: 'bg-white/10 hover:bg-white/20 text-white border-white/15 hover:border-white/25 shadow-lg hover:shadow-xl focus:ring-white/30 rounded-2xl',
      floating: 'bg-white/20 hover:bg-white/30 text-white border-white/25 hover:border-white/35 shadow-2xl hover:shadow-3xl focus:ring-white/30 rounded-full',
      outline: 'bg-transparent hover:bg-white/10 text-white border-white/30 hover:border-white/40 focus:ring-white/30 rounded-2xl',
      ghost: 'bg-transparent hover:bg-white/10 text-white border-transparent hover:border-white/20 focus:ring-white/30 rounded-2xl'
    };

    const sizes = {
      sm: 'px-4 py-2 text-sm',
      md: 'px-6 py-3 text-base',
      lg: 'px-8 py-4 text-lg'
    };

    const combinedClassName = `${baseClasses} ${variants[variant]} ${sizes[size]} ${className}`;

    return (
      <motion.button
        ref={ref}
        className={combinedClassName}
        disabled={disabled || loading}
        whileHover={{ 
          scale: disabled || loading ? 1 : 1.02,
          y: disabled || loading ? 0 : -1
        }}
        whileTap={{ 
          scale: disabled || loading ? 1 : 0.98,
          y: disabled || loading ? 0 : 0
        }}
        transition={{
          type: "spring",
          stiffness: 400,
          damping: 17
        }}
        {...props}
      >
        {loading && (
          <Loader2 className="w-4 h-4 mr-2 animate-spin" />
        )}
        {children}
      </motion.button>
    );
  }
);

Button.displayName = 'Button';

export default Button;