import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  FlatList,
  RefreshControl,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { GlassCard, GlassBadge } from '../components';

interface HistoryItem {
  id: string;
  filename: string;
  upload_date: string;
  pitch_type: string;
  confidence: number;
  processing_time: number;
}

const HistoryScreen: React.FC = () => {
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [refreshing, setRefreshing] = useState(false);

  // Mock data for demonstration
  const mockHistory: HistoryItem[] = [
    {
      id: '1',
      filename: 'pitch_1.mp4',
      upload_date: '2024-01-15T10:30:00Z',
      pitch_type: 'FF',
      confidence: 0.92,
      processing_time: 3.2,
    },
    {
      id: '2',
      filename: 'slider_demo.mp4',
      upload_date: '2024-01-14T15:45:00Z',
      pitch_type: 'SL',
      confidence: 0.88,
      processing_time: 2.8,
    },
    {
      id: '3',
      filename: 'changeup_practice.mp4',
      upload_date: '2024-01-13T09:15:00Z',
      pitch_type: 'CH',
      confidence: 0.85,
      processing_time: 3.5,
    },
  ];

  useEffect(() => {
    loadHistory();
  }, []);

  const loadHistory = async () => {
    // TODO: Load from shared API client
    setHistory(mockHistory);
  };

  const onRefresh = async () => {
    setRefreshing(true);
    await loadHistory();
    setRefreshing(false);
  };

  const formatPitchType = (pitchType: string): string => {
    const pitchTypes: Record<string, string> = {
      'FF': 'Four-Seam Fastball',
      'SL': 'Slider',
      'CH': 'Changeup',
      'CU': 'Curveball',
      'SI': 'Sinker',
      'FC': 'Cutter',
    };
    return pitchTypes[pitchType] || pitchType;
  };

  const formatDate = (dateString: string): string => {
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  const getConfidenceColor = (confidence: number): string => {
    if (confidence >= 0.9) return '#34C759';
    if (confidence >= 0.7) return '#FF9500';
    return '#FF3B30';
  };

  const getConfidenceVariant = (confidence: number): 'success' | 'warning' | 'error' => {
    if (confidence >= 0.9) return 'success';
    if (confidence >= 0.7) return 'warning';
    return 'error';
  };

  const renderHistoryItem = ({ item }: { item: HistoryItem }) => (
    <GlassCard style={styles.historyItem}>
      <View style={styles.itemHeader}>
        <View style={styles.fileInfo}>
          <View style={styles.videoIcon}>
            <Ionicons name="videocam" size={20} color="rgba(255, 255, 255, 0.9)" />
          </View>
          <View style={styles.fileDetails}>
            <Text style={styles.filename} numberOfLines={1}>
              {item.filename}
            </Text>
            <Text style={styles.date}>{formatDate(item.upload_date)}</Text>
          </View>
        </View>
        <GlassBadge 
          text={item.pitch_type} 
          variant={getConfidenceVariant(item.confidence)}
        />
      </View>
      
      <View style={styles.itemContent}>
        <Text style={styles.pitchType}>{formatPitchType(item.pitch_type)}</Text>
        <View style={styles.metrics}>
          <View style={styles.metric}>
            <View style={styles.metricIcon}>
              <Ionicons name="analytics" size={14} color={getConfidenceColor(item.confidence)} />
            </View>
            <Text style={[styles.confidence, { color: getConfidenceColor(item.confidence) }]}>
              {Math.round(item.confidence * 100)}% confidence
            </Text>
          </View>
          <View style={styles.metric}>
            <View style={styles.metricIcon}>
              <Ionicons name="time" size={14} color="rgba(255, 255, 255, 0.6)" />
            </View>
            <Text style={styles.processingTime}>{item.processing_time}s processing</Text>
          </View>
        </View>
      </View>
    </GlassCard>
  );

  const renderEmptyState = () => (
    <View style={styles.emptyState}>
      <View style={styles.emptyIcon}>
        <Ionicons name="list-outline" size={48} color="rgba(255, 255, 255, 0.3)" />
      </View>
      <Text style={styles.emptyTitle}>No Analysis History</Text>
      <Text style={styles.emptySubtitle}>
        Upload your first pitch video to see results here
      </Text>
    </View>
  );

  return (
    <View style={styles.container}>
      <FlatList
        data={history}
        renderItem={renderHistoryItem}
        keyExtractor={(item) => item.id}
        style={styles.list}
        contentContainerStyle={history.length === 0 ? styles.emptyContainer : undefined}
        refreshControl={
          <RefreshControl refreshing={refreshing} onRefresh={onRefresh} />
        }
        ListEmptyComponent={renderEmptyState}
      />
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: 'rgba(0, 0, 0, 1)',
  },
  list: {
    flex: 1,
    padding: 16,
    paddingTop: 60,
  },
  emptyContainer: {
    flex: 1,
    justifyContent: 'center',
  },
  historyItem: {
    padding: 20,
    marginBottom: 12,
  },
  itemHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 16,
  },
  fileInfo: {
    flexDirection: 'row',
    alignItems: 'center',
    flex: 1,
  },
  videoIcon: {
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: 'rgba(255, 255, 255, 0.15)',
    alignItems: 'center',
    justifyContent: 'center',
    marginRight: 12,
  },
  fileDetails: {
    flex: 1,
  },
  filename: {
    fontSize: 16,
    fontWeight: '600',
    color: 'rgba(255, 255, 255, 0.9)',
    marginBottom: 2,
  },
  date: {
    fontSize: 13,
    color: 'rgba(255, 255, 255, 0.6)',
  },
  itemContent: {
    gap: 12,
  },
  pitchType: {
    fontSize: 20,
    fontWeight: '700',
    color: 'rgba(255, 255, 255, 0.9)',
    marginBottom: 8,
  },
  metrics: {
    gap: 8,
  },
  metric: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  metricIcon: {
    width: 24,
    height: 24,
    borderRadius: 12,
    backgroundColor: 'rgba(255, 255, 255, 0.3)',
    alignItems: 'center',
    justifyContent: 'center',
    marginRight: 8,
  },
  confidence: {
    fontSize: 14,
    fontWeight: '500',
  },
  processingTime: {
    fontSize: 14,
    color: 'rgba(255, 255, 255, 0.6)',
  },
  emptyState: {
    alignItems: 'center',
    padding: 40,
  },
  emptyIcon: {
    width: 80,
    height: 80,
    borderRadius: 40,
    backgroundColor: 'rgba(255, 255, 255, 0.3)',
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 20,
  },
  emptyTitle: {
    fontSize: 22,
    fontWeight: '700',
    color: 'rgba(255, 255, 255, 0.9)',
    marginBottom: 8,
  },
  emptySubtitle: {
    fontSize: 16,
    color: 'rgba(255, 255, 255, 0.6)',
    textAlign: 'center',
    lineHeight: 24,
    maxWidth: 280,
  },
});

export default HistoryScreen;