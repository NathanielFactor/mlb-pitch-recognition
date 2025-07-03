import React from 'react';
import { View, Text, StyleSheet, ViewStyle, TextStyle } from 'react-native';
import { Ionicons } from '@expo/vector-icons';

interface GlassBadgeProps {
  text: string;
  variant?: 'default' | 'success' | 'warning' | 'error';
  icon?: keyof typeof Ionicons.glyphMap;
  style?: ViewStyle;
  textStyle?: TextStyle;
}

const GlassBadge: React.FC<GlassBadgeProps> = ({
  text,
  variant = 'default',
  icon,
  style,
  textStyle,
}) => {
  const badgeStyle = [
    styles.base,
    styles[variant],
    style,
  ];

  const textStyles = [
    styles.text,
    styles[`${variant}Text` as keyof typeof styles],
    textStyle,
  ];

  const getIconColor = () => {
    switch (variant) {
      case 'success': return '#34C759';
      case 'warning': return '#FF9500';
      case 'error': return '#FF3B30';
      default: return 'rgba(0, 0, 0, 0.6)';
    }
  };

  return (
    <View style={badgeStyle}>
      {icon && (
        <Ionicons
          name={icon}
          size={12}
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
    backgroundColor: 'rgba(255, 255, 255, 0.6)',
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.2)',
    borderRadius: 9999,
    paddingVertical: 4,
    paddingHorizontal: 8,
    alignSelf: 'flex-start',
  },
  
  // Variants
  default: {
    backgroundColor: 'rgba(255, 255, 255, 0.6)',
    borderColor: 'rgba(255, 255, 255, 0.2)',
  },
  success: {
    backgroundColor: 'rgba(52, 199, 89, 0.2)',
    borderColor: '#34C759',
  },
  warning: {
    backgroundColor: 'rgba(255, 149, 0, 0.2)',
    borderColor: '#FF9500',
  },
  error: {
    backgroundColor: 'rgba(255, 59, 48, 0.2)',
    borderColor: '#FF3B30',
  },
  
  // Text styles
  text: {
    fontSize: 12,
    fontWeight: '600',
    fontFamily: 'system-ui',
  },
  defaultText: {
    color: 'rgba(0, 0, 0, 0.9)',
  },
  successText: {
    color: '#34C759',
  },
  warningText: {
    color: '#FF9500',
  },
  errorText: {
    color: '#FF3B30',
  },
  
  icon: {
    marginRight: 4,
  },
});

export default GlassBadge;