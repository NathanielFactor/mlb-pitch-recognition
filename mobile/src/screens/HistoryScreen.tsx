import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  FlatList,
  TouchableOpacity,
  RefreshControl,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';

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
    if (confidence >= 0.9) return '#10b981';
    if (confidence >= 0.7) return '#f59e0b';
    return '#ef4444';
  };

  const renderHistoryItem = ({ item }: { item: HistoryItem }) => (
    <TouchableOpacity style={styles.historyItem}>
      <View style={styles.itemHeader}>
        <View style={styles.fileInfo}>
          <Ionicons name="videocam" size={20} color="#6b7280" />
          <Text style={styles.filename} numberOfLines={1}>
            {item.filename}
          </Text>
        </View>
        <Text style={styles.date}>{formatDate(item.upload_date)}</Text>
      </View>
      
      <View style={styles.itemContent}>
        <View style={styles.pitchInfo}>
          <Text style={styles.pitchType}>{formatPitchType(item.pitch_type)}</Text>
          <View style={styles.metrics}>
            <View style={styles.metric}>
              <Ionicons name="analytics" size={16} color={getConfidenceColor(item.confidence)} />
              <Text style={[styles.confidence, { color: getConfidenceColor(item.confidence) }]}>
                {Math.round(item.confidence * 100)}%
              </Text>
            </View>
            <View style={styles.metric}>
              <Ionicons name="time" size={16} color="#6b7280" />
              <Text style={styles.processingTime}>{item.processing_time}s</Text>
            </View>
          </View>
        </View>
      </View>
    </TouchableOpacity>
  );

  const renderEmptyState = () => (
    <View style={styles.emptyState}>
      <Ionicons name="list-outline" size={64} color="#d1d5db" />
      <Text style={styles.emptyTitle}>No History Yet</Text>
      <Text style={styles.emptySubtitle}>
        Upload your first pitch video to see analysis results here
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
    backgroundColor: '#f9fafb',
  },
  list: {
    flex: 1,
    padding: 16,
  },
  emptyContainer: {
    flex: 1,
    justifyContent: 'center',
  },
  historyItem: {
    backgroundColor: 'white',
    borderRadius: 12,
    padding: 16,
    marginBottom: 12,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  itemHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 12,
  },
  fileInfo: {
    flexDirection: 'row',
    alignItems: 'center',
    flex: 1,
  },
  filename: {
    fontSize: 16,
    fontWeight: '600',
    color: '#1f2937',
    marginLeft: 8,
    flex: 1,
  },
  date: {
    fontSize: 12,
    color: '#6b7280',
  },
  itemContent: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  pitchInfo: {
    flex: 1,
  },
  pitchType: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#1e40af',
    marginBottom: 8,
  },
  metrics: {
    flexDirection: 'row',
    gap: 16,
  },
  metric: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  confidence: {
    fontSize: 14,
    fontWeight: '600',
    marginLeft: 4,
  },
  processingTime: {
    fontSize: 14,
    color: '#6b7280',
    marginLeft: 4,
  },
  emptyState: {
    alignItems: 'center',
    padding: 40,
  },
  emptyTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#1f2937',
    marginTop: 16,
    marginBottom: 8,
  },
  emptySubtitle: {
    fontSize: 16,
    color: '#6b7280',
    textAlign: 'center',
    lineHeight: 24,
  },
});

export default HistoryScreen;