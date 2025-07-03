// Apple-style animation presets and utilities
// Cross-platform animation configurations

export const appleEasing = {
  // Apple's signature easing curves
  default: 'cubic-bezier(0.4, 0.0, 0.2, 1)',
  easeIn: 'cubic-bezier(0.4, 0.0, 1, 1)',
  easeOut: 'cubic-bezier(0.0, 0.0, 0.2, 1)',
  easeInOut: 'cubic-bezier(0.4, 0.0, 0.2, 1)',
  spring: 'cubic-bezier(0.175, 0.885, 0.32, 1.275)',
  bounce: 'cubic-bezier(0.68, -0.55, 0.265, 1.55)',
} as const;

export const appleTimings = {
  // Apple's preferred timing durations
  fast: 150,
  normal: 250,
  slow: 350,
  slower: 500,
  slowest: 750,
} as const;

// Framer Motion variants for Apple-style interactions
export const appleMotionVariants = {
  // Button interactions
  button: {
    initial: { scale: 1, y: 0 },
    hover: { 
      scale: 1.02, 
      y: -1,
      transition: {
        type: "spring",
        stiffness: 400,
        damping: 17
      }
    },
    tap: { 
      scale: 0.98, 
      y: 0,
      transition: {
        type: "spring",
        stiffness: 600,
        damping: 20
      }
    }
  },

  // Card interactions
  card: {
    initial: { y: 0, scale: 1, rotateX: 0 },
    hover: { 
      y: -4, 
      scale: 1.02,
      rotateX: 5,
      transition: {
        type: "spring",
        stiffness: 300,
        damping: 15
      }
    }
  },

  // Floating elements
  floating: {
    initial: { y: 0 },
    animate: {
      y: [-2, 2, -2],
      transition: {
        duration: 3,
        repeat: Infinity,
        ease: "easeInOut"
      }
    }
  },

  // Page transitions
  page: {
    initial: { 
      opacity: 0, 
      y: 20, 
      scale: 0.95 
    },
    animate: { 
      opacity: 1, 
      y: 0, 
      scale: 1,
      transition: {
        duration: 0.4,
        ease: [0.4, 0.0, 0.2, 1]
      }
    },
    exit: { 
      opacity: 0, 
      y: -20, 
      scale: 0.95,
      transition: {
        duration: 0.3,
        ease: [0.4, 0.0, 1, 1]
      }
    }
  },

  // Modal/overlay
  modal: {
    backdrop: {
      initial: { opacity: 0 },
      animate: { 
        opacity: 1,
        transition: { duration: 0.3 }
      },
      exit: { 
        opacity: 0,
        transition: { duration: 0.2 }
      }
    },
    content: {
      initial: { 
        opacity: 0, 
        scale: 0.9, 
        y: 20 
      },
      animate: { 
        opacity: 1, 
        scale: 1, 
        y: 0,
        transition: {
          type: "spring",
          damping: 25,
          stiffness: 500
        }
      },
      exit: { 
        opacity: 0, 
        scale: 0.9, 
        y: 20,
        transition: { duration: 0.2 }
      }
    }
  },

  // List item stagger
  listContainer: {
    initial: {},
    animate: {
      transition: {
        staggerChildren: 0.1
      }
    }
  },
  listItem: {
    initial: { 
      opacity: 0, 
      x: -20 
    },
    animate: { 
      opacity: 1, 
      x: 0,
      transition: {
        type: "spring",
        stiffness: 400,
        damping: 25
      }
    }
  }
};

// React Native specific animations
export const mobileAnimations = {
  // Spring configuration for React Native
  spring: {
    damping: 15,
    stiffness: 400,
    mass: 1
  },

  // Timing configuration
  timing: {
    duration: appleTimings.normal,
    easing: 'easeInOut' as const
  },

  // Layout animation presets
  layout: {
    spring: {
      type: 'spring' as const,
      damping: 25,
      stiffness: 500,
      mass: 1
    },
    timing: {
      type: 'timing' as const,
      duration: appleTimings.normal
    }
  }
};

// CSS animation keyframes for web
export const cssAnimations = {
  // Apple spring entrance
  appleSpring: `
    @keyframes apple-spring {
      0% { 
        transform: scale(0.95) translateY(4px);
        opacity: 0;
      }
      50% {
        transform: scale(1.02) translateY(-2px);
        opacity: 0.8;
      }
      100% { 
        transform: scale(1) translateY(0px);
        opacity: 1;
      }
    }
  `,

  // Glass floating effect
  glassFloat: `
    @keyframes glass-float {
      0%, 100% { 
        transform: translateY(0px);
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.1);
      }
      50% { 
        transform: translateY(-4px);
        box-shadow: 0 20px 35px -5px rgba(0, 0, 0, 0.15);
      }
    }
  `,

  // Backdrop blur entrance
  backdropBlurIn: `
    @keyframes backdrop-blur-in {
      0% { 
        backdrop-filter: blur(0px);
        background-color: transparent;
      }
      100% { 
        backdrop-filter: blur(16px);
        background-color: rgba(255, 255, 255, 0.8);
      }
    }
  `
};

// Utility functions for animations
export const animationUtils = {
  // Create staggered delay for list items
  getStaggerDelay: (index: number, baseDelay: number = 0.1): number => {
    return index * baseDelay;
  },

  // Spring configuration generator
  createSpring: (stiffness: number = 400, damping: number = 17) => ({
    type: "spring" as const,
    stiffness,
    damping
  }),

  // Timing configuration generator
  createTiming: (duration: number = appleTimings.normal, ease: string = appleEasing.default) => ({
    duration,
    ease
  }),

  // Generate entrance animation
  createEntrance: (delay: number = 0) => ({
    initial: { opacity: 0, y: 20, scale: 0.95 },
    animate: { 
      opacity: 1, 
      y: 0, 
      scale: 1,
      transition: {
        delay,
        ...animationUtils.createSpring()
      }
    }
  })
};