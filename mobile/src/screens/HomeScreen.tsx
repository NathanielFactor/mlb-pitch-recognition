import React from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  ImageBackground,
  Dimensions,
} from 'react-native';
import { useNavigation } from '@react-navigation/native';
import { BottomTabNavigationProp } from '@react-navigation/bottom-tabs';
import { Ionicons } from '@expo/vector-icons';
import { RootTabParamList } from '../navigation/AppNavigator';
import { GlassCard, GlassButton } from '../components';

type HomeScreenNavigationProp = BottomTabNavigationProp<RootTabParamList, 'Home'>;

const { width, height } = Dimensions.get('window');

const HomeScreen: React.FC = () => {
  const navigation = useNavigation<HomeScreenNavigationProp>();

  const navigateToUpload = () => {
    navigation.navigate('Upload');
  };

  return (
    <ScrollView style={styles.container} showsVerticalScrollIndicator={false}>
      <ImageBackground
        source={{ uri: 'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTAwIiBoZWlnaHQ9IjEwMCIgdmlld0JveD0iMCAwIDEwMCAxMDAiIGZpbGw9Im5vbmUiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+CjxkZWZzPgo8bGluZWFyR3JhZGllbnQgaWQ9ImdyYWRpZW50IiB4MT0iMCUiIHkxPSIwJSIgeDI9IjEwMCUiIHkyPSIxMDAlIj4KPHN0b3Agb2Zmc2V0PSIwJSIgc3R5bGU9InN0b3AtY29sb3I6IzAwN0FGRjtzdG9wLW9wYWNpdHk6MC4xIiAvPgo8c3RvcCBvZmZzZXQ9IjEwMCUiIHN0eWxlPSJzdG9wLWNvbG9yOiMzNEM3NTk7c3RvcC1vcGFjaXR5OjAuMSIgLz4KPC9saW5lYXJHcmFkaWVudD4KPC9kZWZzPgo8cmVjdCB3aWR0aD0iMTAwIiBoZWlnaHQ9IjEwMCIgZmlsbD0idXJsKCNncmFkaWVudCkiLz4KPHN2Zz4K' }}
        style={styles.backgroundGradient}
        resizeMode="cover"
      >
        <View style={styles.content}>
          {/* Hero Section with Glass Effect */}
          <GlassCard variant="floating" style={styles.hero}>
            <View style={styles.heroContent}>
              <View style={styles.heroIcon}>
                <Ionicons name="baseball" size={64} color="rgba(255, 255, 255, 0.9)" />
              </View>
              <Text style={styles.title}>Pitch Recognition</Text>
              <Text style={styles.subtitle}>
                AI-powered baseball pitch analysis
              </Text>
              <GlassButton
                title="Start Analysis"
                onPress={navigateToUpload}
                icon="camera"
                variant="primary"
                size="lg"
                style={styles.heroButton}
              />
            </View>
          </GlassCard>

          {/* Features Section */}
          <View style={styles.features}>
            <Text style={styles.sectionTitle}>Capabilities</Text>
            
            <GlassCard style={styles.featureCard}>
              <View style={styles.featureIcon}>
                <Ionicons name="camera" size={24} color="rgba(255, 255, 255, 0.9)" />
              </View>
              <View style={styles.featureText}>
                <Text style={styles.featureTitle}>Record & Upload</Text>
                <Text style={styles.featureDesc}>
                  Capture pitch videos or select from gallery
                </Text>
              </View>
            </GlassCard>

            <GlassCard style={styles.featureCard}>
              <View style={styles.featureIcon}>
                <Ionicons name="analytics" size={24} color="rgba(255, 255, 255, 0.9)" />
              </View>
              <View style={styles.featureText}>
                <Text style={styles.featureTitle}>AI Analysis</Text>
                <Text style={styles.featureDesc}>
                  Advanced ML models analyze pitch mechanics
                </Text>
              </View>
            </GlassCard>

            <GlassCard style={styles.featureCard}>
              <View style={styles.featureIcon}>
                <Ionicons name="list" size={24} color="rgba(255, 255, 255, 0.9)" />
              </View>
              <View style={styles.featureText}>
                <Text style={styles.featureTitle}>Track History</Text>
                <Text style={styles.featureDesc}>
                  View all your pitch analysis history
                </Text>
              </View>
            </GlassCard>
          </View>

          {/* Quick Stats */}
          <GlassCard variant="elevated" style={styles.stats}>
            <Text style={styles.sectionTitle}>Your Progress</Text>
            <View style={styles.statsGrid}>
              <View style={styles.statItem}>
                <Text style={styles.statNumber}>47</Text>
                <Text style={styles.statLabel}>Pitches</Text>
              </View>
              <View style={styles.statItem}>
                <Text style={styles.statNumber}>92%</Text>
                <Text style={styles.statLabel}>Accuracy</Text>
              </View>
              <View style={styles.statItem}>
                <Text style={styles.statNumber}>4</Text>
                <Text style={styles.statLabel}>Types</Text>
              </View>
            </View>
          </GlassCard>
        </View>
      </ImageBackground>
    </ScrollView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: 'rgba(242, 242, 247, 1)',
  },
  backgroundGradient: {
    flex: 1,
    minHeight: height,
  },
  content: {
    padding: 20,
    paddingTop: 60,
  },
  hero: {
    marginBottom: 32,
    minHeight: 280,
  },
  heroContent: {
    alignItems: 'center',
    padding: 32,
  },
  heroIcon: {
    width: 80,
    height: 80,
    borderRadius: 40,
    backgroundColor: 'rgba(255, 255, 255, 0.15)',
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 24,
  },
  title: {
    fontSize: 32,
    fontWeight: '700',
    color: 'rgba(255, 255, 255, 0.9)',
    marginBottom: 12,
    textAlign: 'center',
    letterSpacing: -0.5,
  },
  subtitle: {
    fontSize: 18,
    color: 'rgba(255, 255, 255, 0.7)',
    textAlign: 'center',
    marginBottom: 32,
    lineHeight: 26,
    maxWidth: 280,
  },
  heroButton: {
    minWidth: 180,
  },
  features: {
    marginBottom: 32,
  },
  sectionTitle: {
    fontSize: 24,
    fontWeight: '700',
    color: 'rgba(255, 255, 255, 0.9)',
    marginBottom: 20,
    letterSpacing: -0.3,
  },
  featureCard: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 20,
    marginBottom: 12,
  },
  featureIcon: {
    width: 48,
    height: 48,
    borderRadius: 24,
    backgroundColor: 'rgba(255, 255, 255, 0.15)',
    alignItems: 'center',
    justifyContent: 'center',
    marginRight: 16,
  },
  featureText: {
    flex: 1,
  },
  featureTitle: {
    fontSize: 17,
    fontWeight: '600',
    color: 'rgba(255, 255, 255, 0.9)',
    marginBottom: 4,
  },
  featureDesc: {
    fontSize: 15,
    color: 'rgba(255, 255, 255, 0.7)',
    lineHeight: 22,
  },
  stats: {
    padding: 24,
    marginBottom: 20,
  },
  statsGrid: {
    flexDirection: 'row',
    justifyContent: 'space-around',
  },
  statItem: {
    alignItems: 'center',
  },
  statNumber: {
    fontSize: 28,
    fontWeight: '700',
    color: 'rgba(255, 255, 255, 0.9)',
    marginBottom: 4,
  },
  statLabel: {
    fontSize: 13,
    color: 'rgba(255, 255, 255, 0.7)',
    textAlign: 'center',
    fontWeight: '500',
  },
});

export default HomeScreen;