# Apple Spatial Design System

A comprehensive, cross-platform design system implementing Apple's Spatial Design principles with glass-like elements and depth hierarchy.

## Overview

This design system provides consistent visual language across web and mobile platforms while maintaining Apple's signature glassmorphism effects and spatial depth.

## Core Principles

### 1. Glass Morphism
- Translucent backgrounds with backdrop blur
- Subtle borders and shadows for depth
- Layered transparency for hierarchy

### 2. Spatial Hierarchy
- **Base**: Standard glass surfaces
- **Elevated**: Enhanced depth for important content
- **Floating**: Maximum depth for primary actions

### 3. Apple-Style Interactions
- Spring animations with proper easing
- Subtle micro-interactions
- Consistent touch/hover feedback

## Design Tokens

### Colors
```typescript
// Glass surfaces
colors.glass.primary     // rgba(255, 255, 255, 0.8)
colors.glass.secondary   // rgba(255, 255, 255, 0.6)
colors.glass.tertiary    // rgba(255, 255, 255, 0.4)

// Apple system colors
colors.accent.blue       // #007AFF
colors.accent.green      // #34C759
colors.accent.orange     // #FF9500
colors.accent.red        // #FF3B30
```

### Typography
```typescript
typography.fontFamily.system  // Apple system font stack
typography.fontSize.base      // 16px base size
typography.fontWeight.medium  // 500 weight for UI elements
```

### Spacing
```typescript
spacing.sm   // 8px
spacing.md   // 16px
spacing.lg   // 24px
spacing.xl   // 32px
```

### Shadows
```typescript
shadows.sm   // Subtle depth
shadows.md   // Standard elevation
shadows.lg   // Prominent floating
```

## Components

### Glass Card
```typescript
// Base card with standard glass effect
<GlassCard variant="base">Content</GlassCard>

// Elevated card with enhanced depth
<GlassCard variant="elevated">Content</GlassCard>

// Floating card with maximum depth
<GlassCard variant="floating">Content</GlassCard>
```

### Glass Button
```typescript
// Primary action button
<GlassButton variant="primary" size="lg">
  Start Analysis
</GlassButton>

// Secondary button
<GlassButton variant="secondary" size="md">
  Cancel
</GlassButton>

// Floating action button
<GlassButton variant="floating" size="lg">
  Upload
</GlassButton>
```

### Glass Badge
```typescript
// Success state
<GlassBadge variant="success" text="Complete" />

// Warning state
<GlassBadge variant="warning" text="Processing" />

// Error state
<GlassBadge variant="error" text="Failed" />
```

## Platform Implementation

### Web (React)
Uses CSS backdrop-filter and Tailwind utilities:
```css
.glass {
  background: rgba(255, 255, 255, 0.8);
  backdrop-filter: blur(16px);
  border: 1px solid rgba(255, 255, 255, 0.2);
}
```

### Mobile (React Native)
Uses platform-specific styling with similar visual effects:
```typescript
const styles = StyleSheet.create({
  glass: {
    backgroundColor: 'rgba(255, 255, 255, 0.8)',
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.2)',
    shadowColor: 'rgba(0, 0, 0, 0.15)',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 1,
    shadowRadius: 8,
    elevation: 4,
  },
});
```

## Animations

### Apple Spring
Signature entrance animation with spring physics:
```typescript
// Framer Motion (Web)
whileHover={{ 
  scale: 1.02, 
  y: -1,
  transition: {
    type: "spring",
    stiffness: 400,
    damping: 17
  }
}}

// React Native
Animated.spring(animatedValue, {
  toValue: 1,
  damping: 15,
  stiffness: 400,
  mass: 1,
  useNativeDriver: true,
})
```

### Glass Float
Subtle floating animation for elevated elements:
```css
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
```

## Usage Guidelines

### When to Use Glass Effects
- ✅ Navigation elements
- ✅ Modal overlays
- ✅ Floating action buttons
- ✅ Card-based layouts
- ❌ Dense data tables
- ❌ Form inputs (use subtle effects)

### Accessibility
- Ensure sufficient contrast ratios
- Provide alternative styling for reduced transparency preferences
- Test with VoiceOver/screen readers
- Support dynamic type scaling

### Performance
- Use backdrop-filter sparingly on mobile
- Prefer CSS transforms over layout changes
- Implement proper will-change hints
- Test on lower-end devices

## Color Psychology

### Blue (#007AFF)
- Primary actions
- System elements
- Trust and reliability

### Green (#34C759)
- Success states
- Positive feedback
- Growth and progress

### Orange (#FF9500)
- Warning states
- Attention-grabbing
- Energy and enthusiasm

### Red (#FF3B30)
- Error states
- Destructive actions
- Urgency and importance

## Migration Guide

### From Solid to Glass
1. Replace solid backgrounds with translucent equivalents
2. Add backdrop-filter where supported
3. Update shadows to spatial depth system
4. Implement spring animations
5. Test accessibility and performance

### Component Updates
```typescript
// Before
<div className="bg-white shadow-lg">

// After
<div className="bg-white/80 backdrop-blur-md shadow-lg border border-white/20">
```

## Browser Support

### Backdrop Filter
- ✅ Safari 14+
- ✅ Chrome 76+
- ✅ Firefox 103+
- ⚠️ Fallback: solid backgrounds

### Performance Considerations
- Use `will-change: backdrop-filter` for animations
- Limit blur radius to 24px maximum
- Avoid nesting multiple backdrop-filter elements

## Contributing

When adding new components:
1. Follow glass morphism principles
2. Ensure cross-platform compatibility
3. Include accessibility considerations
4. Provide usage examples
5. Test on multiple devices/browsers