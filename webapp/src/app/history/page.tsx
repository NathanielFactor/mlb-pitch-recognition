'use client';

import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { FileVideo, Calendar, Clock, Target, Search, Download, Eye, TrendingUp } from 'lucide-react';
import { Button, Card, Badge, ProgressBar } from '@/components/ui';

interface HistoryItem {
  id: string;
  filename: string;
  upload_date: string;
  pitch_type: string;
  confidence: number;
  processing_time: number;
  status: 'completed' | 'failed' | 'processing';
}

export default function HistoryPage() {
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedPitchType, setSelectedPitchType] = useState<string>('all');
  const [sortBy, setSortBy] = useState<'date' | 'confidence' | 'processing_time'>('date');
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('desc');

  // Fetch history data from API
  useEffect(() => {
    const fetchHistory = async () => {
      try {
        setLoading(true);
        setError(null);
        
        // This will be replaced with real API call when history endpoint is available
        const response = await fetch('/api/history').catch(() => {
          // Fallback to mock data if API is not available
          return {
            json: () => Promise.resolve({
              items: [
                {
                  id: '1',
                  filename: 'fastball_pitch_1.mp4',
                  upload_date: '2024-06-30T14:30:00Z',
                  pitch_type: 'Fastball',
                  confidence: 0.92,
                  processing_time: 2.1,
                  status: 'completed'
                },
                {
                  id: '2', 
                  filename: 'curveball_sample.mp4',
                  upload_date: '2024-06-29T16:45:00Z',
                  pitch_type: 'Curveball',
                  confidence: 0.87,
                  processing_time: 2.5,
                  status: 'completed'
                },
                {
                  id: '3',
                  filename: 'slider_demo.mp4',
                  upload_date: '2024-06-28T12:20:00Z',
                  pitch_type: 'Slider',
                  confidence: 0.79,
                  processing_time: 1.9,
                  status: 'completed'
                },
                {
                  id: '4',
                  filename: 'changeup_example.mov',
                  upload_date: '2024-06-27T18:15:00Z',
                  pitch_type: 'Changeup',
                  confidence: 0.85,
                  processing_time: 2.3,
                  status: 'completed'
                },
                {
                  id: '5',
                  filename: 'knuckleball_rare.mp4',
                  upload_date: '2024-06-26T09:30:00Z',
                  pitch_type: 'Knuckleball',
                  confidence: 0.73,
                  processing_time: 3.1,
                  status: 'completed'
                }
              ]
            })
          };
        });
        
        const data = await response.json();
        setHistory(data.items || []);
      } catch (err) {
        setError('Failed to load prediction history');
        console.error('Error fetching history:', err);
      } finally {
        setLoading(false);
      }
    };

    fetchHistory();
  }, []);

  // Get unique pitch types for filter
  const pitchTypes = Array.from(new Set(history.map(item => item.pitch_type)));

  // Filter and sort history
  const filteredHistory = history
    .filter(item => {
      const matchesSearch = item.filename.toLowerCase().includes(searchTerm.toLowerCase()) ||
                          item.pitch_type.toLowerCase().includes(searchTerm.toLowerCase());
      const matchesPitchType = selectedPitchType === 'all' || item.pitch_type === selectedPitchType;
      return matchesSearch && matchesPitchType && item.status === 'completed';
    })
    .sort((a, b) => {
      let aValue, bValue;
      
      switch (sortBy) {
        case 'date':
          aValue = new Date(a.upload_date).getTime();
          bValue = new Date(b.upload_date).getTime();
          break;
        case 'confidence':
          aValue = a.confidence;
          bValue = b.confidence;
          break;
        case 'processing_time':
          aValue = a.processing_time;
          bValue = b.processing_time;
          break;
        default:
          return 0;
      }
      
      return sortOrder === 'asc' ? aValue - bValue : bValue - aValue;
    });

  // Calculate statistics
  const stats = {
    total: history.filter(item => item.status === 'completed').length,
    avgConfidence: history.length > 0 
      ? (history.reduce((acc, item) => acc + item.confidence, 0) / history.length * 100).toFixed(1)
      : '0',
    avgProcessingTime: history.length > 0
      ? (history.reduce((acc, item) => acc + item.processing_time, 0) / history.length).toFixed(1)
      : '0',
    uniquePitchTypes: pitchTypes.length
  };

  const getPitchTypeColor = (pitchType: string) => {
    const colors = {
      'Fastball': 'bg-red-500/20 text-red-300 border-red-400/30',
      'Curveball': 'bg-purple-500/20 text-purple-300 border-purple-400/30',
      'Slider': 'bg-blue-500/20 text-blue-300 border-blue-400/30',
      'Changeup': 'bg-green-500/20 text-green-300 border-green-400/30',
      'Knuckleball': 'bg-orange-500/20 text-orange-300 border-orange-400/30',
      'Cutter': 'bg-gray-500/20 text-gray-300 border-gray-400/30'
    };
    return colors[pitchType as keyof typeof colors] || 'bg-white/15 text-white border-white/25';
  };

  const getPitchTypeIcon = (pitchType: string) => {
    const icons = {
      'Fastball': '⚡',
      'Curveball': '🌪️',
      'Slider': '↗️',
      'Changeup': '🐌',
      'Knuckleball': '🤏',
      'Cutter': '✂️'
    };
    return icons[pitchType as keyof typeof icons] || '⚾';
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-black">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        {/* Header */}
        <motion.div 
          className="mb-8"
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
        >
          <h1 className="text-4xl font-bold text-white mb-2">Prediction History</h1>
          <p className="text-xl text-white/70">
            Track your pitch analysis journey with detailed insights and analytics
          </p>
        </motion.div>

        {/* Statistics Cards */}
        <motion.div 
          className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.1 }}
        >
          <Card className="text-center p-6">
            <div className="w-12 h-12 bg-blue-500/20 rounded-full flex items-center justify-center mx-auto mb-3">
              <TrendingUp className="w-6 h-6 text-blue-400" />
            </div>
            <div className="text-3xl font-bold text-white mb-1">{stats.total}</div>
            <div className="text-sm text-white/70">Total Predictions</div>
          </Card>
          
          <Card className="text-center p-6">
            <div className="w-12 h-12 bg-green-500/20 rounded-full flex items-center justify-center mx-auto mb-3">
              <Target className="w-6 h-6 text-green-400" />
            </div>
            <div className="text-3xl font-bold text-white mb-1">{stats.avgConfidence}%</div>
            <div className="text-sm text-white/70">Avg Confidence</div>
          </Card>
          
          <Card className="text-center p-6">
            <div className="w-12 h-12 bg-purple-500/20 rounded-full flex items-center justify-center mx-auto mb-3">
              <Clock className="w-6 h-6 text-purple-400" />
            </div>
            <div className="text-3xl font-bold text-white mb-1">{stats.avgProcessingTime}s</div>
            <div className="text-sm text-white/70">Avg Processing</div>
          </Card>
          
          <Card className="text-center p-6">
            <div className="w-12 h-12 bg-orange-500/20 rounded-full flex items-center justify-center mx-auto mb-3">
              <FileVideo className="w-6 h-6 text-orange-400" />
            </div>
            <div className="text-3xl font-bold text-white mb-1">{stats.uniquePitchTypes}</div>
            <div className="text-sm text-white/70">Pitch Types</div>
          </Card>
        </motion.div>

        {/* Filters and Search */}
        <motion.div 
          className="mb-6"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.2 }}
        >
          <Card className="p-6">
            <div className="flex flex-col md:flex-row gap-4 items-center">
              {/* Search */}
              <div className="relative flex-1">
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-white/50" />
                <input
                  type="text"
                  placeholder="Search by filename or pitch type..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  className="w-full pl-10 pr-4 py-2 bg-white/10 border border-white/20 rounded-lg text-white placeholder-white/50 focus:outline-none focus:ring-2 focus:ring-blue-500/50"
                />
              </div>
              
              {/* Pitch Type Filter */}
              <select
                value={selectedPitchType}
                onChange={(e) => setSelectedPitchType(e.target.value)}
                className="px-4 py-2 bg-white/10 border border-white/20 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-blue-500/50"
              >
                <option value="all">All Pitch Types</option>
                {pitchTypes.map(type => (
                  <option key={type} value={type}>{type}</option>
                ))}
              </select>
              
              {/* Sort */}
              <select
                value={`${sortBy}-${sortOrder}`}
                onChange={(e) => {
                  const [field, order] = e.target.value.split('-');
                  setSortBy(field as 'date' | 'confidence' | 'processing_time');
                  setSortOrder(order as 'asc' | 'desc');
                }}
                className="px-4 py-2 bg-white/10 border border-white/20 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-blue-500/50"
              >
                <option value="date-desc">Newest First</option>
                <option value="date-asc">Oldest First</option>
                <option value="confidence-desc">Highest Confidence</option>
                <option value="confidence-asc">Lowest Confidence</option>
                <option value="processing_time-asc">Fastest Processing</option>
                <option value="processing_time-desc">Slowest Processing</option>
              </select>
            </div>
          </Card>
        </motion.div>

        {/* Loading State */}
        {loading && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="text-center py-12"
          >
            <div className="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-white"></div>
            <p className="mt-4 text-white/70">Loading prediction history...</p>
          </motion.div>
        )}

        {/* Error State */}
        {error && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="text-center py-12"
          >
            <div className="text-red-400 mb-4">⚠️ {error}</div>
            <Button onClick={() => window.location.reload()}>
              Try Again
            </Button>
          </motion.div>
        )}

        {/* Empty State */}
        {!loading && !error && filteredHistory.length === 0 && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="text-center py-12"
          >
            <div className="w-16 h-16 bg-white/10 rounded-full flex items-center justify-center mx-auto mb-4">
              <FileVideo className="w-8 h-8 text-white/50" />
            </div>
            <h3 className="text-xl font-medium text-white mb-2">
              {history.length === 0 ? 'No predictions yet' : 'No results found'}
            </h3>
            <p className="text-white/70 mb-6">
              {history.length === 0 
                ? 'Get started by uploading your first pitch video.'
                : 'Try adjusting your search or filter criteria.'
              }
            </p>
            <Button 
              onClick={() => window.location.href = '/upload'}
              size="lg"
            >
              <FileVideo className="w-5 h-5 mr-2" />
              Upload Video
            </Button>
          </motion.div>
        )}

        {/* History List */}
        {!loading && !error && filteredHistory.length > 0 && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.3 }}
          >
            <Card className="overflow-hidden">
              <div className="divide-y divide-white/10">
                {filteredHistory.map((item, index) => (
                  <motion.div
                    key={item.id}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ duration: 0.3, delay: index * 0.05 }}
                    className="p-6 hover:bg-white/5 transition-colors"
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex items-center space-x-4">
                        {/* Pitch Type Icon */}
                        <div className="w-12 h-12 bg-white/10 rounded-full flex items-center justify-center">
                          <span className="text-xl">{getPitchTypeIcon(item.pitch_type)}</span>
                        </div>
                        
                        {/* Details */}
                        <div className="flex-1">
                          <div className="flex items-center gap-3 mb-2">
                            <h3 className="text-lg font-medium text-white">{item.filename}</h3>
                            <Badge 
                              className={`${getPitchTypeColor(item.pitch_type)} border`}
                              size="sm"
                            >
                              {item.pitch_type}
                            </Badge>
                          </div>
                          <div className="flex items-center gap-6 text-sm text-white/70">
                            <div className="flex items-center gap-1">
                              <Calendar className="w-4 h-4" />
                              {new Date(item.upload_date).toLocaleDateString()}
                            </div>
                            <div className="flex items-center gap-1">
                              <Target className="w-4 h-4" />
                              {(item.confidence * 100).toFixed(1)}% confidence
                            </div>
                            <div className="flex items-center gap-1">
                              <Clock className="w-4 h-4" />
                              {item.processing_time}s processing
                            </div>
                          </div>
                        </div>
                      </div>
                      
                      {/* Actions */}
                      <div className="flex items-center gap-2">
                        <Button variant="outline" size="sm">
                          <Eye className="w-4 h-4 mr-1" />
                          View
                        </Button>
                        <Button variant="outline" size="sm">
                          <Download className="w-4 h-4" />
                        </Button>
                      </div>
                    </div>
                    
                    {/* Confidence Bar */}
                    <div className="mt-4">
                      <div className="flex items-center justify-between text-xs text-white/50 mb-1">
                        <span>Confidence Score</span>
                        <span>{(item.confidence * 100).toFixed(1)}%</span>
                      </div>
                      <ProgressBar 
                        progress={item.confidence * 100} 
                        variant="success" 
                        size="sm"
                        showPercentage={false}
                      />
                    </div>
                  </motion.div>
                ))}
              </div>
            </Card>
          </motion.div>
        )}
      </div>
    </div>
  );
}