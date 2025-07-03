import React, { useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  Alert,
  ScrollView,
  Dimensions,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import * as ImagePicker from 'expo-image-picker';
import { Camera } from 'expo-camera';
import { GlassCard, GlassButton, GlassBadge } from '../components';

const UploadScreen: React.FC = () => {
  const [selectedVideo, setSelectedVideo] = useState<string | null>(null);
  const [isUploading, setIsUploading] = useState(false);

  const requestPermissions = async () => {
    const { status: cameraStatus } = await Camera.requestCameraPermissionsAsync();
    const { status: mediaStatus } = await ImagePicker.requestMediaLibraryPermissionsAsync();
    
    if (cameraStatus !== 'granted' || mediaStatus !== 'granted') {
      Alert.alert(
        'Permissions Required',
        'Please grant camera and media library permissions to use this feature.'
      );
      return false;
    }
    return true;
  };

  const recordVideo = async () => {
    const hasPermissions = await requestPermissions();
    if (!hasPermissions) return;

    try {
      const result = await ImagePicker.launchCameraAsync({
        mediaTypes: ImagePicker.MediaTypeOptions.Videos,
        allowsEditing: true,
        quality: 0.8,
        videoMaxDuration: 30, // 30 seconds max
      });

      if (!result.canceled && result.assets[0]) {
        setSelectedVideo(result.assets[0].uri);
      }
    } catch (error) {
      Alert.alert('Error', 'Failed to record video');
    }
  };

  const selectFromGallery = async () => {
    const hasPermissions = await requestPermissions();
    if (!hasPermissions) return;

    try {
      const result = await ImagePicker.launchImageLibraryAsync({
        mediaTypes: ImagePicker.MediaTypeOptions.Videos,
        allowsEditing: true,
        quality: 0.8,
      });

      if (!result.canceled && result.assets[0]) {
        setSelectedVideo(result.assets[0].uri);
      }
    } catch (error) {
      Alert.alert('Error', 'Failed to select video');
    }
  };

  const uploadVideo = async () => {
    if (!selectedVideo) {
      Alert.alert('No Video', 'Please select or record a video first');
      return;
    }

    setIsUploading(true);
    
    // TODO: Implement actual upload using shared API client
    try {
      await new Promise(resolve => setTimeout(resolve, 2000)); // Mock upload
      Alert.alert('Success', 'Video uploaded successfully!');
      setSelectedVideo(null);
    } catch (error) {
      Alert.alert('Error', 'Failed to upload video');
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <ScrollView style={styles.container} showsVerticalScrollIndicator={false}>
      <View style={styles.content}>
        <View style={styles.header}>
          <Text style={styles.title}>Upload Pitch Video</Text>
          <Text style={styles.subtitle}>
            Record a new video or select from your gallery
          </Text>
        </View>

        <View style={styles.actions}>
          <GlassCard style={styles.actionCard}>
            <GlassButton
              title="Record Video"
              onPress={recordVideo}
              icon="camera"
              variant="floating"
              size="lg"
              style={styles.actionButton}
            />
            <Text style={styles.actionDescription}>
              Capture a pitch in real-time
            </Text>
          </GlassCard>

          <GlassCard style={styles.actionCard}>
            <GlassButton
              title="Select from Gallery"
              onPress={selectFromGallery}
              icon="images"
              variant="secondary"
              size="lg"
              style={styles.actionButton}
            />
            <Text style={styles.actionDescription}>
              Choose from existing videos
            </Text>
          </GlassCard>
        </View>

        {selectedVideo && (
          <GlassCard variant="elevated" style={styles.selectedVideo}>
            <View style={styles.videoPreview}>
              <View style={styles.videoIcon}>
                <Ionicons name="videocam" size={32} color="rgba(255, 255, 255, 0.9)" />
              </View>
              <View style={styles.videoInfo}>
                <Text style={styles.videoSelectedText}>Video Ready</Text>
                <Text style={styles.videoPath} numberOfLines={1} ellipsizeMode="middle">
                  {selectedVideo.split('/').pop()}
                </Text>
              </View>
              <GlassBadge text="Ready" variant="success" icon="checkmark-circle" />
            </View>
            
            <GlassButton
              title={isUploading ? 'Uploading...' : 'Analyze Video'}
              onPress={uploadVideo}
              icon={isUploading ? 'sync' : 'cloud-upload'}
              variant="primary"
              size="lg"
              disabled={isUploading}
              style={styles.uploadButton}
            />
          </GlassCard>
        )}

        <GlassCard style={styles.tips}>
          <Text style={styles.tipsTitle}>Tips for Best Results</Text>
          <View style={styles.tipsList}>
            <View style={styles.tip}>
              <View style={styles.tipIcon}>
                <Ionicons name="camera" size={16} color="rgba(255, 255, 255, 0.9)" />
              </View>
              <Text style={styles.tipText}>Record from the side angle</Text>
            </View>
            <View style={styles.tip}>
              <View style={styles.tipIcon}>
                <Ionicons name="hand-left" size={16} color="rgba(255, 255, 255, 0.9)" />
              </View>
              <Text style={styles.tipText}>Keep the camera steady</Text>
            </View>
            <View style={styles.tip}>
              <View style={styles.tipIcon}>
                <Ionicons name="sunny" size={16} color="rgba(255, 255, 255, 0.9)" />
              </View>
              <Text style={styles.tipText}>Ensure good lighting</Text>
            </View>
            <View style={styles.tip}>
              <View style={styles.tipIcon}>
                <Ionicons name="fitness" size={16} color="rgba(255, 255, 255, 0.9)" />
              </View>
              <Text style={styles.tipText}>Capture the full pitch motion</Text>
            </View>
          </View>
        </GlassCard>
      </View>
    </ScrollView>
  );
};

const { width } = Dimensions.get('window');

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: 'rgba(242, 242, 247, 1)',
  },
  content: {
    padding: 20,
    paddingTop: 60,
  },
  header: {
    alignItems: 'center',
    marginBottom: 32,
  },
  title: {
    fontSize: 28,
    fontWeight: '700',
    color: 'rgba(255, 255, 255, 0.9)',
    marginBottom: 8,
    letterSpacing: -0.3,
  },
  subtitle: {
    fontSize: 17,
    color: 'rgba(255, 255, 255, 0.7)',
    textAlign: 'center',
    lineHeight: 24,
    maxWidth: 280,
  },
  actions: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 24,
    gap: 12,
  },
  actionCard: {
    flex: 1,
    alignItems: 'center',
    padding: 20,
    minHeight: 140,
    justifyContent: 'space-between',
  },
  actionButton: {
    width: '100%',
    marginBottom: 12,
  },
  actionDescription: {
    fontSize: 13,
    color: 'rgba(255, 255, 255, 0.7)',
    textAlign: 'center',
    lineHeight: 18,
  },
  selectedVideo: {
    padding: 24,
    marginBottom: 24,
  },
  videoPreview: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 20,
  },
  videoIcon: {
    width: 48,
    height: 48,
    borderRadius: 24,
    backgroundColor: 'rgba(255, 255, 255, 0.15)',
    alignItems: 'center',
    justifyContent: 'center',
    marginRight: 16,
  },
  videoInfo: {
    flex: 1,
  },
  videoSelectedText: {
    fontSize: 17,
    fontWeight: '600',
    color: 'rgba(255, 255, 255, 0.9)',
    marginBottom: 4,
  },
  videoPath: {
    fontSize: 14,
    color: 'rgba(255, 255, 255, 0.7)',
    maxWidth: '100%',
  },
  uploadButton: {
    width: '100%',
  },
  tips: {
    padding: 20,
  },
  tipsTitle: {
    fontSize: 20,
    fontWeight: '600',
    color: 'rgba(255, 255, 255, 0.9)',
    marginBottom: 16,
  },
  tipsList: {
    gap: 12,
  },
  tip: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  tipIcon: {
    width: 32,
    height: 32,
    borderRadius: 16,
    backgroundColor: 'rgba(255, 255, 255, 0.15)',
    alignItems: 'center',
    justifyContent: 'center',
    marginRight: 12,
  },
  tipText: {
    fontSize: 15,
    color: 'rgba(255, 255, 255, 0.8)',
    lineHeight: 20,
    flex: 1,
  },
});

export default UploadScreen;