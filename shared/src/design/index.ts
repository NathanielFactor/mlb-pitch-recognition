// Apple Spatial Design System
// Export all design tokens and component styles

export * from './tokens';
export * from './components';
export * from './animations';

// Theme configurations for different platforms
export const theme = {
  light: {
    background: 'rgba(242, 242, 247, 1)',
    surface: 'rgba(255, 255, 255, 0.8)',
    text: 'rgba(0, 0, 0, 0.9)',
    glass: 'rgba(255, 255, 255, 0.8)',
    border: 'rgba(0, 0, 0, 0.1)',
  },
  dark: {
    background: 'rgba(0, 0, 0, 1)',
    surface: 'rgba(28, 28, 30, 0.8)',
    text: 'rgba(255, 255, 255, 0.9)',
    glass: 'rgba(28, 28, 30, 0.8)',
    border: 'rgba(255, 255, 255, 0.1)',
  },
} as const;