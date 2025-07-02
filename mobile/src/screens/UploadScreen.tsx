import React, { useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  Alert,
  ScrollView,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import * as ImagePicker from 'expo-image-picker';
import { Camera } from 'expo-camera';

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
    <ScrollView style={styles.container}>
      <View style={styles.content}>
        <Text style={styles.title}>Upload Pitch Video</Text>
        <Text style={styles.subtitle}>
          Record a new video or select from your gallery
        </Text>

        <View style={styles.actions}>
          <TouchableOpacity style={styles.actionButton} onPress={recordVideo}>
            <Ionicons name="camera" size={32} color="#1e40af" />
            <Text style={styles.actionText}>Record Video</Text>
          </TouchableOpacity>

          <TouchableOpacity style={styles.actionButton} onPress={selectFromGallery}>
            <Ionicons name="images" size={32} color="#1e40af" />
            <Text style={styles.actionText}>Select from Gallery</Text>
          </TouchableOpacity>
        </View>

        {selectedVideo && (
          <View style={styles.selectedVideo}>
            <Ionicons name="videocam" size={48} color="#10b981" />
            <Text style={styles.videoSelectedText}>Video Selected</Text>
            <Text style={styles.videoPath} numberOfLines={1} ellipsizeMode="middle">
              {selectedVideo}
            </Text>
            
            <TouchableOpacity
              style={[styles.uploadButton, isUploading && styles.uploadButtonDisabled]}
              onPress={uploadVideo}
              disabled={isUploading}
            >
              {isUploading ? (
                <Ionicons name="sync" size={24} color="white" />
              ) : (
                <Ionicons name="cloud-upload" size={24} color="white" />
              )}
              <Text style={styles.uploadButtonText}>
                {isUploading ? 'Uploading...' : 'Upload Video'}
              </Text>
            </TouchableOpacity>
          </View>
        )}

        <View style={styles.tips}>
          <Text style={styles.tipsTitle}>Tips for Best Results</Text>
          <View style={styles.tip}>
            <Ionicons name="checkmark-circle" size={20} color="#10b981" />
            <Text style={styles.tipText}>Record from the side angle</Text>
          </View>
          <View style={styles.tip}>
            <Ionicons name="checkmark-circle" size={20} color="#10b981" />
            <Text style={styles.tipText}>Keep the camera steady</Text>
          </View>
          <View style={styles.tip}>
            <Ionicons name="checkmark-circle" size={20} color="#10b981" />
            <Text style={styles.tipText}>Ensure good lighting</Text>
          </View>
          <View style={styles.tip}>
            <Ionicons name="checkmark-circle" size={20} color="#10b981" />
            <Text style={styles.tipText}>Capture the full pitch motion</Text>
          </View>
        </View>
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
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#1f2937',
    marginBottom: 8,
  },
  subtitle: {
    fontSize: 16,
    color: '#6b7280',
    marginBottom: 32,
    lineHeight: 24,
  },
  actions: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 32,
  },
  actionButton: {
    flex: 1,
    backgroundColor: 'white',
    alignItems: 'center',
    padding: 24,
    borderRadius: 12,
    marginHorizontal: 8,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  actionText: {
    fontSize: 16,
    fontWeight: '600',
    color: '#1f2937',
    marginTop: 12,
    textAlign: 'center',
  },
  selectedVideo: {
    backgroundColor: 'white',
    padding: 24,
    borderRadius: 12,
    alignItems: 'center',
    marginBottom: 32,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  videoSelectedText: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#10b981',
    marginTop: 12,
  },
  videoPath: {
    fontSize: 12,
    color: '#6b7280',
    marginTop: 4,
    marginBottom: 20,
    maxWidth: '100%',
  },
  uploadButton: {
    backgroundColor: '#1e40af',
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 12,
    paddingHorizontal: 24,
    borderRadius: 8,
  },
  uploadButtonDisabled: {
    opacity: 0.6,
  },
  uploadButtonText: {
    color: 'white',
    fontSize: 16,
    fontWeight: 'bold',
    marginLeft: 8,
  },
  tips: {
    backgroundColor: 'white',
    padding: 20,
    borderRadius: 12,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  tipsTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#1f2937',
    marginBottom: 16,
  },
  tip: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 12,
  },
  tipText: {
    fontSize: 14,
    color: '#4b5563',
    marginLeft: 12,
  },
});

export default UploadScreen;