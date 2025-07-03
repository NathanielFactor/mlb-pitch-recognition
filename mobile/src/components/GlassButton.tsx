import React from 'react';
import { TouchableOpacity, Text, StyleSheet, ViewStyle, TextStyle } from 'react-native';
import { Ionicons } from '@expo/vector-icons';

interface GlassButtonProps {
  title: string;
  onPress: () => void;
  icon?: keyof typeof Ionicons.glyphMap;
  variant?: 'primary' | 'secondary' | 'floating';
  size?: 'sm' | 'md' | 'lg';
  disabled?: boolean;
  style?: ViewStyle;
  textStyle?: TextStyle;
}

const GlassButton: React.FC<GlassButtonProps> = ({
  title,
  onPress,
  icon,
  variant = 'primary',
  size = 'md',
  disabled = false,
  style,
  textStyle,
}) => {
  const buttonStyle = [
    styles.base,
    styles[variant],
    styles[size],
    disabled && styles.disabled,
    style,
  ];

  const textStyles = [
    styles.text,
    styles[`${variant}Text` as keyof typeof styles],
    styles[`${size}Text` as keyof typeof styles],
    disabled && styles.disabledText,
    textStyle,
  ];

  return (
    <TouchableOpacity
      style={buttonStyle}
      onPress={onPress}
      disabled={disabled}
      activeOpacity={0.8}
    >
      {icon && (
        <Ionicons
          name={icon}
          size={size === 'sm' ? 16 : size === 'lg' ? 24 : 20}
          color={variant === 'primary' ? '#007AFF' : 'rgba(0, 0, 0, 0.9)'}
          style={styles.icon}
        />
      )}
      <Text style={textStyles}>{title}</Text>
    </TouchableOpacity>
  );
};

const styles = StyleSheet.create({
  base: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    borderWidth: 1,
    shadowColor: 'rgba(0, 0, 0, 0.1)',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 1,
    shadowRadius: 2,
    elevation: 2,
  },
  
  // Variants
  primary: {
    backgroundColor: 'rgba(0, 122, 255, 0.2)',
    borderColor: '#007AFF',
    borderRadius: 12,
  },
  secondary: {
    backgroundColor: 'rgba(255, 255, 255, 0.8)',
    borderColor: 'rgba(255, 255, 255, 0.2)',
    borderRadius: 12,
  },
  floating: {
    backgroundColor: 'rgba(255, 255, 255, 0.6)',
    borderColor: 'rgba(255, 255, 255, 0.2)',
    borderRadius: 9999,
    shadowColor: 'rgba(0, 0, 0, 0.15)',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 1,
    shadowRadius: 8,
    elevation: 4,
  },
  
  // Sizes
  sm: {
    paddingVertical: 8,
    paddingHorizontal: 16,
  },
  md: {
    paddingVertical: 12,
    paddingHorizontal: 24,
  },
  lg: {
    paddingVertical: 16,
    paddingHorizontal: 32,
  },
  
  // Disabled state
  disabled: {
    opacity: 0.5,
  },
  
  // Text styles
  text: {
    fontFamily: 'system-ui',
    fontWeight: '600',
    textAlign: 'center',
  },
  primaryText: {
    color: '#007AFF',
    fontSize: 16,
  },
  secondaryText: {
    color: 'rgba(0, 0, 0, 0.9)',
    fontSize: 16,
  },
  floatingText: {
    color: 'rgba(0, 0, 0, 0.9)',
    fontSize: 16,
  },
  
  // Size-specific text
  smText: {
    fontSize: 14,
  },
  mdText: {
    fontSize: 16,
  },
  lgText: {
    fontSize: 18,
  },
  
  disabledText: {
    opacity: 0.6,
  },
  
  icon: {
    marginRight: 8,
  },
});

export default GlassButton;