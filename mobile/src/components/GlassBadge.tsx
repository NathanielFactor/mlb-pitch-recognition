import React from 'react';
import { View, Text, StyleSheet, ViewStyle, TextStyle } from 'react-native';
import { Ionicons } from '@expo/vector-icons';

interface GlassBadgeProps {
  text: string;
  variant?: 'fastball' | 'curveball' | 'slider' | 'changeup' | 'knuckleball' | 'cutter' | 'success' | 'warning' | 'error' | 'default';
  size?: 'sm' | 'md' | 'lg';
  icon?: keyof typeof Ionicons.glyphMap;
  style?: ViewStyle;
  textStyle?: TextStyle;
}

const GlassBadge: React.FC<GlassBadgeProps> = ({
  text,
  variant = 'default',
  size = 'md',
  icon,
  style,
  textStyle,
}) => {
  const badgeStyle = [
    styles.base,
    styles[variant],
    styles[size],
    style,
  ];

  const textStyles = [
    styles.text,
    styles[`${variant}Text` as keyof typeof styles],
    styles[`${size}Text` as keyof typeof styles],
    textStyle,
  ];

  const getIconColor = () => {
    switch (variant) {
      case 'success': return 'rgba(255, 255, 255, 0.9)';
      case 'warning': return 'rgba(255, 255, 255, 0.9)';
      case 'error': return 'rgba(255, 255, 255, 0.9)';
      default: return 'rgba(255, 255, 255, 0.9)';
    }
  };

  return (
    <View style={badgeStyle}>
      {icon && (
        <Ionicons
          name={icon}
          size={size === 'sm' ? 10 : size === 'lg' ? 14 : 12}
          color={getIconColor()}
          style={styles.icon}
        />
      )}
      <Text style={textStyles}>{text}</Text>
    </View>
  );
};

const styles = StyleSheet.create({
  base: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    fontWeight: '500',
    borderRadius: 9999,
    alignSelf: 'flex-start',
    backgroundColor: 'rgba(255, 255, 255, 0.15)',
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.25)',
  },
  
  // Variants
  fastball: {
    backgroundColor: 'rgba(255, 255, 255, 0.15)',
    borderColor: 'rgba(255, 255, 255, 0.25)',
  },
  curveball: {
    backgroundColor: 'rgba(255, 255, 255, 0.15)',
    borderColor: 'rgba(255, 255, 255, 0.25)',
  },
  slider: {
    backgroundColor: 'rgba(255, 255, 255, 0.15)',
    borderColor: 'rgba(255, 255, 255, 0.25)',
  },
  changeup: {
    backgroundColor: 'rgba(255, 255, 255, 0.15)',
    borderColor: 'rgba(255, 255, 255, 0.25)',
  },
  knuckleball: {
    backgroundColor: 'rgba(255, 255, 255, 0.15)',
    borderColor: 'rgba(255, 255, 255, 0.25)',
  },
  cutter: {
    backgroundColor: 'rgba(255, 255, 255, 0.15)',
    borderColor: 'rgba(255, 255, 255, 0.25)',
  },
  success: {
    backgroundColor: 'rgba(255, 255, 255, 0.15)',
    borderColor: 'rgba(255, 255, 255, 0.25)',
  },
  warning: {
    backgroundColor: 'rgba(255, 255, 255, 0.15)',
    borderColor: 'rgba(255, 255, 255, 0.25)',
  },
  error: {
    backgroundColor: 'rgba(255, 255, 255, 0.15)',
    borderColor: 'rgba(255, 255, 255, 0.25)',
  },
  default: {
    backgroundColor: 'rgba(255, 255, 255, 0.15)',
    borderColor: 'rgba(255, 255, 255, 0.25)',
  },
  
  // Sizes
  sm: {
    paddingVertical: 2,
    paddingHorizontal: 8,
  },
  md: {
    paddingVertical: 4,
    paddingHorizontal: 10,
  },
  lg: {
    paddingVertical: 6,
    paddingHorizontal: 12,
  },
  
  // Text styles
  text: {
    fontWeight: '500',
    fontFamily: 'system-ui',
  },
  fastballText: {
    color: 'rgba(255, 255, 255, 0.9)',
  },
  curveballText: {
    color: 'rgba(255, 255, 255, 0.9)',
  },
  sliderText: {
    color: 'rgba(255, 255, 255, 0.9)',
  },
  changeupText: {
    color: 'rgba(255, 255, 255, 0.9)',
  },
  knuckleballText: {
    color: 'rgba(255, 255, 255, 0.9)',
  },
  cutterText: {
    color: 'rgba(255, 255, 255, 0.9)',
  },
  successText: {
    color: 'rgba(255, 255, 255, 0.9)',
  },
  warningText: {
    color: 'rgba(255, 255, 255, 0.9)',
  },
  errorText: {
    color: 'rgba(255, 255, 255, 0.9)',
  },
  defaultText: {
    color: 'rgba(255, 255, 255, 0.9)',
  },
  
  // Size-specific text
  smText: {
    fontSize: 12,
  },
  mdText: {
    fontSize: 14,
  },
  lgText: {
    fontSize: 16,
  },
  
  icon: {
    marginRight: 4,
  },
});

export default GlassBadge;