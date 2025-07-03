// Apple Spatial Design Component Styles
// Reusable component patterns for glass-like UI elements

import { colors, shadows, glassmorphism, borderRadius, spacing } from './tokens';

export const glassCard = {
  base: {
    backgroundColor: colors.glass.primary,
    backdropFilter: glassmorphism.backdropFilter.md,
    borderWidth: glassmorphism.border.width,
    borderStyle: glassmorphism.border.style,
    borderColor: colors.border.glass,
    borderRadius: borderRadius.xl,
    ...shadows.md,
  },
  elevated: {
    backgroundColor: colors.glass.secondary,
    backdropFilter: glassmorphism.backdropFilter.lg,
    borderWidth: glassmorphism.border.width,
    borderStyle: glassmorphism.border.style,
    borderColor: colors.border.glass,
    borderRadius: borderRadius.xl,
    ...shadows.lg,
  },
  floating: {
    backgroundColor: colors.glass.tertiary,
    backdropFilter: glassmorphism.backdropFilter.xl,
    borderWidth: glassmorphism.border.width,
    borderStyle: glassmorphism.border.style,
    borderColor: colors.border.glass,
    borderRadius: borderRadius['2xl'],
    ...shadows.xl,
  },
} as const;

export const glassButton = {
  primary: {
    backgroundColor: colors.accent.blueGlass,
    backdropFilter: glassmorphism.backdropFilter.md,
    borderWidth: glassmorphism.border.width,
    borderStyle: glassmorphism.border.style,
    borderColor: colors.accent.blue,
    borderRadius: borderRadius.lg,
    paddingVertical: spacing.sm,
    paddingHorizontal: spacing.md,
    ...shadows.sm,
  },
  secondary: {
    backgroundColor: colors.glass.primary,
    backdropFilter: glassmorphism.backdropFilter.md,
    borderWidth: glassmorphism.border.width,
    borderStyle: glassmorphism.border.style,
    borderColor: colors.border.glass,
    borderRadius: borderRadius.lg,
    paddingVertical: spacing.sm,
    paddingHorizontal: spacing.md,
    ...shadows.sm,
  },
  floating: {
    backgroundColor: colors.glass.secondary,
    backdropFilter: glassmorphism.backdropFilter.lg,
    borderWidth: glassmorphism.border.width,
    borderStyle: glassmorphism.border.style,
    borderColor: colors.border.glass,
    borderRadius: borderRadius.full,
    paddingVertical: spacing.md,
    paddingHorizontal: spacing.lg,
    ...shadows.md,
  },
} as const;

export const glassNavigation = {
  tabBar: {
    backgroundColor: colors.glass.primary,
    backdropFilter: glassmorphism.backdropFilter.lg,
    borderTopWidth: glassmorphism.border.width,
    borderTopColor: colors.border.glass,
    borderTopLeftRadius: borderRadius.xl,
    borderTopRightRadius: borderRadius.xl,
    paddingTop: spacing.sm,
    paddingBottom: spacing.md,
    paddingHorizontal: spacing.md,
    ...shadows.lg,
  },
  sidebar: {
    backgroundColor: colors.glass.secondary,
    backdropFilter: glassmorphism.backdropFilter.xl,
    borderRightWidth: glassmorphism.border.width,
    borderRightColor: colors.border.glass,
    paddingVertical: spacing.lg,
    paddingHorizontal: spacing.md,
    ...shadows.md,
  },
} as const;

export const glassInput = {
  base: {
    backgroundColor: colors.glass.tertiary,
    backdropFilter: glassmorphism.backdropFilter.sm,
    borderWidth: glassmorphism.border.width,
    borderStyle: glassmorphism.border.style,
    borderColor: colors.border.glass,
    borderRadius: borderRadius.lg,
    paddingVertical: spacing.sm,
    paddingHorizontal: spacing.md,
    color: colors.text.primary,
  },
  focused: {
    backgroundColor: colors.glass.secondary,
    backdropFilter: glassmorphism.backdropFilter.md,
    borderColor: colors.accent.blue,
    ...shadows.sm,
  },
} as const;

export const glassModal = {
  backdrop: {
    backgroundColor: 'rgba(0, 0, 0, 0.3)',
    backdropFilter: glassmorphism.backdropFilter.sm,
  },
  content: {
    backgroundColor: colors.glass.primary,
    backdropFilter: glassmorphism.backdropFilter.xl,
    borderWidth: glassmorphism.border.width,
    borderStyle: glassmorphism.border.style,
    borderColor: colors.border.glass,
    borderRadius: borderRadius['2xl'],
    padding: spacing.xl,
    margin: spacing.lg,
    ...shadows.xl,
  },
} as const;

export const glassBadge = {
  default: {
    backgroundColor: colors.glass.secondary,
    backdropFilter: glassmorphism.backdropFilter.sm,
    borderWidth: glassmorphism.border.width,
    borderStyle: glassmorphism.border.style,
    borderColor: colors.border.glass,
    borderRadius: borderRadius.full,
    paddingVertical: spacing.xs,
    paddingHorizontal: spacing.sm,
  },
  success: {
    backgroundColor: colors.accent.greenGlass,
    borderColor: colors.accent.green,
  },
  warning: {
    backgroundColor: colors.accent.orangeGlass,
    borderColor: colors.accent.orange,
  },
  error: {
    backgroundColor: colors.accent.redGlass,
    borderColor: colors.accent.red,
  },
} as const;