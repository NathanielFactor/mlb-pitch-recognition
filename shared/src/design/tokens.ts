// Spatial Design System Tokens
// Glass-like elements with depth and translucency

export const colors = {
  // Glass Surface Colors (translucent approach)
  glass: {
    primary: 'rgba(255, 255, 255, 0.8)',
    secondary: 'rgba(255, 255, 255, 0.6)',
    tertiary: 'rgba(255, 255, 255, 0.4)',
    overlay: 'rgba(255, 255, 255, 0.2)',
  },
  
  // Dark Glass Variants
  darkGlass: {
    primary: 'rgba(28, 28, 30, 0.8)',
    secondary: 'rgba(28, 28, 30, 0.6)',
    tertiary: 'rgba(28, 28, 30, 0.4)',
    overlay: 'rgba(28, 28, 30, 0.2)',
  },
  
  // System Colors (semantic approach)
  system: {
    background: {
      primary: 'rgba(242, 242, 247, 1)',
      secondary: 'rgba(255, 255, 255, 0.8)',
      tertiary: 'rgba(242, 242, 247, 0.6)',
    },
    darkBackground: {
      primary: 'rgba(0, 0, 0, 1)',
      secondary: 'rgba(28, 28, 30, 0.8)',
      tertiary: 'rgba(44, 44, 46, 0.6)',
    },
  },
  
  // Text Colors (spatial style - white text on glass)
  text: {
    primary: 'rgba(255, 255, 255, 0.9)',
    secondary: 'rgba(255, 255, 255, 0.7)',
    tertiary: 'rgba(255, 255, 255, 0.5)',
    inverse: 'rgba(0, 0, 0, 0.9)',
    inverseSecondary: 'rgba(0, 0, 0, 0.6)',
    inverseTertiary: 'rgba(0, 0, 0, 0.4)',
  },
  
  // Accent Colors (spatial style - unified white glass)
  accent: {
    blue: 'rgba(255, 255, 255, 0.9)',
    blueGlass: 'rgba(255, 255, 255, 0.15)',
    green: 'rgba(255, 255, 255, 0.9)',
    greenGlass: 'rgba(255, 255, 255, 0.15)',
    orange: 'rgba(255, 255, 255, 0.9)',
    orangeGlass: 'rgba(255, 255, 255, 0.15)',
    red: 'rgba(255, 255, 255, 0.9)',
    redGlass: 'rgba(255, 255, 255, 0.15)',
  },
  
  // Border Colors
  border: {
    primary: 'rgba(0, 0, 0, 0.1)',
    secondary: 'rgba(0, 0, 0, 0.05)',
    glass: 'rgba(255, 255, 255, 0.2)',
    darkGlass: 'rgba(255, 255, 255, 0.1)',
  },
} as const;

export const spacing = {
  // 8pt grid system with spatial considerations
  xs: 4,
  sm: 8,
  md: 16,
  lg: 24,
  xl: 32,
  '2xl': 48,
  '3xl': 64,
  '4xl': 80,
  '5xl': 96,
} as const;

export const typography = {
  // Type scale approach
  fontFamily: {
    system: 'system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
    mono: 'ui-monospace, SFMono-Regular, "SF Mono", Consolas, monospace',
  },
  fontSize: {
    xs: 12,
    sm: 14,
    base: 16,
    lg: 18,
    xl: 20,
    '2xl': 24,
    '3xl': 30,
    '4xl': 36,
    '5xl': 48,
    '6xl': 60,
    '7xl': 72,
  },
  fontWeight: {
    normal: 400,
    medium: 500,
    semibold: 600,
    bold: 700,
  },
  lineHeight: {
    tight: 1.25,
    normal: 1.5,
    relaxed: 1.75,
  },
} as const;

export const shadows = {
  // Signature depth shadows
  sm: {
    shadowColor: 'rgba(0, 0, 0, 0.1)',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 1,
    shadowRadius: 2,
    elevation: 2,
  },
  md: {
    shadowColor: 'rgba(0, 0, 0, 0.15)',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 1,
    shadowRadius: 8,
    elevation: 4,
  },
  lg: {
    shadowColor: 'rgba(0, 0, 0, 0.2)',
    shadowOffset: { width: 0, height: 8 },
    shadowOpacity: 1,
    shadowRadius: 16,
    elevation: 8,
  },
  xl: {
    shadowColor: 'rgba(0, 0, 0, 0.25)',
    shadowOffset: { width: 0, height: 16 },
    shadowOpacity: 1,
    shadowRadius: 24,
    elevation: 12,
  },
} as const;

export const glassmorphism = {
  // Glassmorphism effects
  backdropFilter: {
    sm: 'blur(8px)',
    md: 'blur(16px)',
    lg: 'blur(24px)',
    xl: 'blur(32px)',
  },
  border: {
    width: 1,
    style: 'solid',
  },
} as const;

export const animations = {
  // Signature easing curves
  easing: {
    default: 'cubic-bezier(0.4, 0.0, 0.2, 1)',
    easeIn: 'cubic-bezier(0.4, 0.0, 1, 1)',
    easeOut: 'cubic-bezier(0.0, 0.0, 0.2, 1)',
    easeInOut: 'cubic-bezier(0.4, 0.0, 0.2, 1)',
    spring: 'cubic-bezier(0.175, 0.885, 0.32, 1.275)',
  },
  duration: {
    fast: 150,
    normal: 250,
    slow: 350,
    slower: 500,
  },
} as const;

export const borderRadius = {
  // Modern radius scale
  none: 0,
  sm: 4,
  md: 8,
  lg: 12,
  xl: 16,
  '2xl': 24,
  '3xl': 32,
  full: 9999,
} as const;

export const zIndex = {
  // Spatial layering system
  base: 0,
  raised: 10,
  floating: 20,
  modal: 30,
  overlay: 40,
  toast: 50,
} as const;