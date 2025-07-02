import React from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
} from 'react-native';
import { useNavigation } from '@react-navigation/native';
import { BottomTabNavigationProp } from '@react-navigation/bottom-tabs';
import { Ionicons } from '@expo/vector-icons';
import { RootTabParamList } from '../navigation/AppNavigator';

type HomeScreenNavigationProp = BottomTabNavigationProp<RootTabParamList, 'Home'>;

const HomeScreen: React.FC = () => {
  const navigation = useNavigation<HomeScreenNavigationProp>();

  return (
    <ScrollView style={styles.container}>
      <View style={styles.content}>
        <View style={styles.hero}>
          <Ionicons name="baseball" size={80} color="#1e40af" />
          <Text style={styles.title}>MLB Pitch Recognition</Text>
          <Text style={styles.subtitle}>
            Classify pitch types from video clips using AI
          </Text>
        </View>

        <View style={styles.features}>
          <Text style={styles.sectionTitle}>Features</Text>
          
          <View style={styles.featureItem}>
            <Ionicons name="camera" size={24} color="#1e40af" />
            <View style={styles.featureText}>
              <Text style={styles.featureTitle}>Record & Upload</Text>
              <Text style={styles.featureDesc}>
                Record pitch videos or upload from gallery
              </Text>
            </View>
          </View>

          <View style={styles.featureItem}>
            <Ionicons name="analytics" size={24} color="#1e40af" />
            <View style={styles.featureText}>
              <Text style={styles.featureTitle}>AI Analysis</Text>
              <Text style={styles.featureDesc}>
                Advanced ML models analyze pitch mechanics
              </Text>
            </View>
          </View>

          <View style={styles.featureItem}>
            <Ionicons name="list" size={24} color="#1e40af" />
            <View style={styles.featureText}>
              <Text style={styles.featureTitle}>Track History</Text>
              <Text style={styles.featureDesc}>
                View all your pitch analysis history
              </Text>
            </View>
          </View>
        </View>

        <TouchableOpacity
          style={styles.uploadButton}
          onPress={() => navigation.navigate('Upload')}
        >
          <Ionicons name="cloud-upload" size={24} color="white" />
          <Text style={styles.uploadButtonText}>Upload Video</Text>
        </TouchableOpacity>
      </View>
    </ScrollView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f9fafb',
  },
  content: {
    padding: 20,
  },
  hero: {
    alignItems: 'center',
    marginBottom: 40,
    paddingVertical: 20,
  },
  title: {
    fontSize: 28,
    fontWeight: 'bold',
    color: '#1f2937',
    marginTop: 16,
    textAlign: 'center',
  },
  subtitle: {
    fontSize: 16,
    color: '#6b7280',
    marginTop: 8,
    textAlign: 'center',
    lineHeight: 24,
  },
  features: {
    marginBottom: 40,
  },
  sectionTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#1f2937',
    marginBottom: 20,
  },
  featureItem: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: 'white',
    padding: 16,
    borderRadius: 12,
    marginBottom: 12,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  featureText: {
    marginLeft: 16,
    flex: 1,
  },
  featureTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#1f2937',
    marginBottom: 4,
  },
  featureDesc: {
    fontSize: 14,
    color: '#6b7280',
    lineHeight: 20,
  },
  uploadButton: {
    backgroundColor: '#1e40af',
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    padding: 16,
    borderRadius: 12,
    shadowColor: '#1e40af',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3,
    shadowRadius: 8,
    elevation: 6,
  },
  uploadButtonText: {
    color: 'white',
    fontSize: 18,
    fontWeight: 'bold',
    marginLeft: 8,
  },
});

export default HomeScreen;