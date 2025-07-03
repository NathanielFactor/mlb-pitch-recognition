import React from 'react';
import { View, StyleSheet, ViewStyle } from 'react-native';

interface GlassCardProps {
  children: React.ReactNode;
  variant?: 'base' | 'elevated' | 'floating';
  style?: ViewStyle;
  hover?: boolean;
  glow?: boolean;
}

const GlassCard: React.FC<GlassCardProps> = ({ 
  children, 
  variant = 'base', 
  style,
  hover = true,
  glow = false
}) => {
  const combinedStyles = [
    styles.base,
    styles[variant],
    glow && styles.glow,
    style
  ];

  return (
    <View style={combinedStyles}>
      {children}
    </View>
  );
};

const styles = StyleSheet.create({
  base: {
    backgroundColor: 'rgba(255, 255, 255, 0.12)',
    borderRadius: 16,
    shadowColor: 'rgba(0, 0, 0, 0.1)',
    shadowOffset: { width: 0, height: 10 },
    shadowOpacity: 1,
    shadowRadius: 25,
    elevation: 4,
    borderColor: 'rgba(255, 255, 255, 0.2)',
    borderWidth: 1,
    overflow: 'hidden',
  },
  elevated: {
    backgroundColor: 'rgba(255, 255, 255, 0.18)',
    borderRadius: 16,
    shadowColor: 'rgba(0, 0, 0, 0.1)',
    shadowOffset: { width: 0, height: 20 },
    shadowOpacity: 1,
    shadowRadius: 25,
    elevation: 8,
    borderColor: 'rgba(255, 255, 255, 0.25)',
  },
  floating: {
    backgroundColor: 'rgba(255, 255, 255, 0.25)',
    borderRadius: 24,
    shadowColor: 'rgba(0, 0, 0, 0.25)',
    shadowOffset: { width: 0, height: 25 },
    shadowOpacity: 1,
    shadowRadius: 50,
    elevation: 12,
    borderColor: 'rgba(255, 255, 255, 0.3)',
  },
  glow: {
    shadowColor: 'rgba(255, 255, 255, 0.25)',
    shadowOffset: { width: 0, height: 0 },
    shadowOpacity: 1,
    shadowRadius: 20,
  },
});

export default GlassCard;