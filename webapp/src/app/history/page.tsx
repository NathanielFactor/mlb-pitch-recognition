export default function HistoryPage() {
  // Mock data for now
  const mockHistory = [
    {
      id: '1',
      filename: 'fastball_pitch_1.mp4',
      upload_date: '2024-06-30',
      pitch_type: 'Fastball',
      confidence: 0.92,
      processing_time: 2.1
    },
    {
      id: '2',
      filename: 'curveball_sample.mp4',
      upload_date: '2024-06-29',
      pitch_type: 'Curveball',
      confidence: 0.87,
      processing_time: 2.5
    },
    {
      id: '3',
      filename: 'slider_demo.mp4',
      upload_date: '2024-06-28',
      pitch_type: 'Slider',
      confidence: 0.79,
      processing_time: 1.9
    },
    {
      id: '4',
      filename: 'changeup_example.mov',
      upload_date: '2024-06-27',
      pitch_type: 'Changeup',
      confidence: 0.85,
      processing_time: 2.3
    }
  ];

  const getPitchTypeColor = (pitchType: string) => {
    return 'bg-white/15 text-white border border-white/25';
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-black">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-white">Prediction History</h1>
          <p className="mt-2 text-white/70">
            View all your previous pitch analysis results
          </p>
        </div>

        {mockHistory.length === 0 ? (
          <div className="text-center py-12">
            <svg className="mx-auto h-12 w-12 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 4V2a1 1 0 011-1h8a1 1 0 011 1v2m0 0V1a1 1 0 011-1h2a1 1 0 011 1v18a1 1 0 01-1 1H4a1 1 0 01-1-1V1a1 1 0 011-1h2a1 1 0 011 1v3m0 0h8m-8 0V7a1 1 0 011-1h6a1 1 0 011 1v7" />
            </svg>
            <h3 className="mt-2 text-sm font-medium text-white">No predictions yet</h3>
            <p className="mt-1 text-sm text-white/70">Get started by uploading your first pitch video.</p>
            <div className="mt-6">
              <a
                href="/upload"
                className="inline-flex items-center px-4 py-2 border border-white/25 shadow-sm text-sm font-medium rounded-md text-white bg-white/15 hover:bg-white/20 backdrop-blur-md"
              >
                Upload Video
              </a>
            </div>
          </div>
        ) : (
          <div className="bg-white/10 backdrop-blur-md border border-white/20 shadow overflow-hidden sm:rounded-2xl">
            <ul className="divide-y divide-white/20">
              {mockHistory.map((item) => (
                <li key={item.id}>
                  <div className="px-4 py-4 sm:px-6">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center">
                        <div className="flex-shrink-0">
                          <svg className="h-8 w-8 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
                          </svg>
                        </div>
                        <div className="ml-4">
                          <div className="flex items-center">
                            <p className="text-sm font-medium text-white truncate">
                              {item.filename}
                            </p>
                            <span className={`ml-2 inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium backdrop-blur-md ${getPitchTypeColor(item.pitch_type)}`}>
                              {item.pitch_type}
                            </span>
                          </div>
                          <div className="mt-1 flex items-center text-sm text-white/70">
                            <p>
                              Uploaded on {new Date(item.upload_date).toLocaleDateString()}
                            </p>
                            <span className="mx-2">•</span>
                            <p>
                              {(item.confidence * 100).toFixed(1)}% confidence
                            </p>
                            <span className="mx-2">•</span>
                            <p>
                              {item.processing_time}s processing
                            </p>
                          </div>
                        </div>
                      </div>
                      <div className="flex items-center space-x-2">
                        <button className="text-white/80 hover:text-white text-sm font-medium">
                          View Details
                        </button>
                        <button className="text-white/50 hover:text-white/70 text-sm font-medium">
                          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 5v.01M12 12v.01M12 19v.01M12 6a1 1 0 110-2 1 1 0 010 2zm0 7a1 1 0 110-2 1 1 0 010 2zm0 7a1 1 0 110-2 1 1 0 010 2z" />
                          </svg>
                        </button>
                      </div>
                    </div>
                  </div>
                </li>
              ))}
            </ul>
          </div>
        )}

        <div className="mt-8 bg-white/10 backdrop-blur-md border border-white/20 rounded-2xl p-6">
          <h3 className="text-lg font-medium text-white mb-2">Statistics Summary</h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="text-center">
              <div className="text-2xl font-bold text-white">{mockHistory.length}</div>
              <div className="text-sm text-white/70">Total Predictions</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-white">
                {mockHistory.length > 0 ? (mockHistory.reduce((acc, item) => acc + item.confidence, 0) / mockHistory.length * 100).toFixed(1) : 0}%
              </div>
              <div className="text-sm text-white/70">Avg Confidence</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-white">
                {mockHistory.length > 0 ? (mockHistory.reduce((acc, item) => acc + item.processing_time, 0) / mockHistory.length).toFixed(1) : 0}s
              </div>
              <div className="text-sm text-white/70">Avg Processing</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-white">
                {Array.from(new Set(mockHistory.map(item => item.pitch_type))).length}
              </div>
              <div className="text-sm text-white/70">Pitch Types</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}