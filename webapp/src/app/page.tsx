'use client';

import Link from 'next/link';
import { motion } from 'framer-motion';
import { Upload, History as HistoryIcon, Zap, Target } from 'lucide-react';
import { Button, Card } from '@/components/ui';

const containerVariants = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: {
      staggerChildren: 0.2,
      delayChildren: 0.1
    }
  }
};

const itemVariants = {
  hidden: { y: 20, opacity: 0 },
  visible: { y: 0, opacity: 1 }
};

export default function Home() {
  return (
    <div className="relative overflow-hidden">
      {/* Hero Section with Glass Background */}
      <section className="relative bg-gradient-to-br from-gray-900 via-gray-800 to-black text-white">
        
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-20">
          <motion.div
            className="text-center"
            variants={containerVariants}
            initial="hidden"
            animate="visible"
          >
            <motion.h1 
              className="text-5xl md:text-7xl font-bold mb-6"
              variants={itemVariants}
              transition={{ duration: 0.6 }}
            >
              <span className="bg-gradient-to-r from-white via-white to-white/80 bg-clip-text text-transparent">
                MLB Pitch
              </span>
              <br />
              <span className="text-white/90">Recognition</span>
            </motion.h1>

            <motion.p 
              className="text-xl md:text-2xl mb-8 max-w-4xl mx-auto text-white/70"
              variants={itemVariants}
            >
              Harness the power of advanced AI to analyze baseball pitches with professional-grade accuracy. 
              Upload your videos and discover the science behind every throw.
            </motion.p>

            <motion.div 
              className="flex flex-col sm:flex-row gap-4 justify-center items-center"
              variants={itemVariants}
            >
              <Link href="/upload">
                <Button size="lg" className="text-lg px-8 py-4">
                  <Upload className="w-5 h-5 mr-2" />
                  Start Analysis
                </Button>
              </Link>
              <Link href="/history">
                <Button variant="outline" size="lg" className="text-lg px-8 py-4">
                  <HistoryIcon className="w-5 h-5 mr-2" />
                  View History
                </Button>
              </Link>
            </motion.div>

            {/* Stats Section */}
            <motion.div 
              className="grid grid-cols-1 md:grid-cols-3 gap-6 mt-16 max-w-4xl mx-auto"
              variants={itemVariants}
            >
              <div className="text-center bg-white/15 backdrop-blur-md rounded-2xl p-6 border border-white/25">
                <div className="text-3xl font-bold text-white">95%+</div>
                <div className="text-white/70">Accuracy Rate</div>
              </div>
              <div className="text-center bg-white/15 backdrop-blur-md rounded-2xl p-6 border border-white/25">
                <div className="text-3xl font-bold text-white">&lt;3s</div>
                <div className="text-white/70">Processing Time</div>
              </div>
              <div className="text-center bg-white/15 backdrop-blur-md rounded-2xl p-6 border border-white/25">
                <div className="text-3xl font-bold text-white">6</div>
                <div className="text-white/70">Pitch Types</div>
              </div>
            </motion.div>
          </motion.div>
        </div>

        {/* Glass overlay */}
        <div className="absolute bottom-0 left-0 right-0 h-20 bg-gradient-to-t from-black/20 to-transparent" />
      </section>

      {/* Features Section */}
      <section className="py-20 bg-gradient-to-b from-gray-900 to-black">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            className="text-center mb-16"
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            viewport={{ once: true }}
          >
            <h2 className="text-4xl font-bold text-white mb-4">
              Professional-Grade Analysis
            </h2>
            <p className="text-xl text-white/70 max-w-3xl mx-auto">
              Our advanced AI system combines computer vision and machine learning to deliver 
              insights that rival professional scouting analysis.
            </p>
          </motion.div>

          <div className="grid grid-cols-1 gap-8 md:grid-cols-2 lg:grid-cols-3">
            <motion.div
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.1 }}
              viewport={{ once: true }}
            >
              <Card hover className="h-full">
                <div className="p-8">
                  <div className="flex items-center mb-6">
                    <div className="w-12 h-12 bg-white/15 rounded-2xl flex items-center justify-center backdrop-blur-md border border-white/20">
                      <Upload className="w-6 h-6 text-white" />
                    </div>
                    <div className="ml-4">
                      <h3 className="text-xl font-bold text-white">Easy Upload</h3>
                      <p className="text-white/70 font-medium">Drag & Drop</p>
                    </div>
                  </div>
                  <p className="text-white/70 leading-relaxed">
                    Simply drag and drop your pitch videos or browse to select files. 
                    Supports MP4, AVI, and MOV formats with automatic preprocessing.
                  </p>
                </div>
              </Card>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.2 }}
              viewport={{ once: true }}
            >
              <Card hover className="h-full">
                <div className="p-8">
                  <div className="flex items-center mb-6">
                    <div className="w-12 h-12 bg-white/15 rounded-2xl flex items-center justify-center backdrop-blur-md border border-white/20">
                      <Zap className="w-6 h-6 text-white" />
                    </div>
                    <div className="ml-4">
                      <h3 className="text-xl font-bold text-white">AI Analysis</h3>
                      <p className="text-white/70 font-medium">CNN + LSTM</p>
                    </div>
                  </div>
                  <p className="text-white/70 leading-relaxed">
                    Advanced neural networks analyze pitch mechanics, ball movement, 
                    and trajectory patterns to identify pitch types with exceptional accuracy.
                  </p>
                </div>
              </Card>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.3 }}
              viewport={{ once: true }}
            >
              <Card hover className="h-full">
                <div className="p-8">
                  <div className="flex items-center mb-6">
                    <div className="w-12 h-12 bg-white/15 rounded-2xl flex items-center justify-center backdrop-blur-md border border-white/20">
                      <Target className="w-6 h-6 text-white" />
                    </div>
                    <div className="ml-4">
                      <h3 className="text-xl font-bold text-white">Instant Results</h3>
                      <p className="text-white/70 font-medium">Real-time</p>
                    </div>
                  </div>
                  <p className="text-white/70 leading-relaxed">
                    Get instant pitch classification with detailed confidence scores, 
                    processing time metrics, and comprehensive analysis reports.
                  </p>
                </div>
              </Card>
            </motion.div>
          </div>

          {/* Supported Pitch Types */}
          <motion.div
            className="mt-20 text-center"
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            viewport={{ once: true }}
          >
            <h3 className="text-2xl font-bold text-white mb-8">Supported Pitch Types</h3>
            <div className="flex flex-wrap justify-center gap-3">
              {['Fastball', 'Curveball', 'Slider', 'Changeup', 'Knuckleball', 'Cutter'].map((pitch, index) => (
                <motion.div
                  key={pitch}
                  className="px-4 py-2 bg-white/15 text-white rounded-full font-medium backdrop-blur-md border border-white/20"
                  initial={{ opacity: 0, scale: 0.8 }}
                  whileInView={{ opacity: 1, scale: 1 }}
                  transition={{ duration: 0.4, delay: index * 0.1 }}
                  viewport={{ once: true }}
                >
                  {pitch}
                </motion.div>
              ))}
            </div>
          </motion.div>
        </div>
      </section>
    </div>
  );
}
