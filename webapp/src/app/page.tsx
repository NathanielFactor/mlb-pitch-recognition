'use client';

import Link from 'next/link';
import { motion } from 'framer-motion';
import { Upload, History as HistoryIcon, Zap, Target, Award } from 'lucide-react';
import { Button, Card, BaseballIcon } from '@/components/ui';

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
  visible: {
    y: 0,
    opacity: 1,
    transition: { duration: 0.6, ease: "easeOut" }
  }
};

export default function Home() {
  return (
    <div className="relative overflow-hidden">
      {/* Hero Section with Baseball Field Background */}
      <section className="relative bg-gradient-to-br from-green-600 via-green-500 to-emerald-600 text-white">
        {/* Animated Baseball */}
        <div className="absolute top-20 right-10 opacity-20">
          <BaseballIcon size={200} spinning />
        </div>
        
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
            >
              <span className="bg-gradient-to-r from-white via-blue-100 to-white bg-clip-text text-transparent">
                MLB Pitch
              </span>
              <br />
              <span className="text-red-400">Recognition</span>
            </motion.h1>

            <motion.p 
              className="text-xl md:text-2xl mb-8 max-w-4xl mx-auto text-green-100"
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
                <Button size="lg" className="bg-gradient-to-r from-red-600 to-red-700 hover:from-red-700 hover:to-red-800 shadow-2xl text-lg px-8 py-4">
                  <Upload className="w-5 h-5 mr-2" />
                  Start Analysis
                </Button>
              </Link>
              <Link href="/history">
                <Button variant="outline" size="lg" className="border-2 border-white text-white hover:bg-white hover:text-green-600 text-lg px-8 py-4">
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
              <div className="text-center bg-white/10 backdrop-blur-md rounded-xl p-6 border border-white/20">
                <div className="text-3xl font-bold text-white">95%+</div>
                <div className="text-green-100">Accuracy Rate</div>
              </div>
              <div className="text-center bg-white/10 backdrop-blur-md rounded-xl p-6 border border-white/20">
                <div className="text-3xl font-bold text-white">&lt;3s</div>
                <div className="text-green-100">Processing Time</div>
              </div>
              <div className="text-center bg-white/10 backdrop-blur-md rounded-xl p-6 border border-white/20">
                <div className="text-3xl font-bold text-white">6</div>
                <div className="text-green-100">Pitch Types</div>
              </div>
            </motion.div>
          </motion.div>
        </div>

        {/* Baseball field lines */}
        <div className="absolute bottom-0 left-0 right-0 h-20 bg-gradient-to-t from-green-800 to-transparent opacity-50" />
      </section>

      {/* Features Section */}
      <section className="py-20 bg-gradient-to-b from-gray-50 to-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            className="text-center mb-16"
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            viewport={{ once: true }}
          >
            <h2 className="text-4xl font-bold text-gray-900 mb-4">
              Professional-Grade Analysis
            </h2>
            <p className="text-xl text-gray-600 max-w-3xl mx-auto">
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
              <Card hover gradient className="h-full">
                <div className="p-8">
                  <div className="flex items-center mb-6">
                    <div className="w-12 h-12 bg-gradient-to-r from-blue-500 to-blue-600 rounded-xl flex items-center justify-center">
                      <Upload className="w-6 h-6 text-white" />
                    </div>
                    <div className="ml-4">
                      <h3 className="text-xl font-bold text-gray-900">Easy Upload</h3>
                      <p className="text-blue-600 font-medium">Drag & Drop</p>
                    </div>
                  </div>
                  <p className="text-gray-600 leading-relaxed">
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
              <Card hover gradient className="h-full">
                <div className="p-8">
                  <div className="flex items-center mb-6">
                    <div className="w-12 h-12 bg-gradient-to-r from-green-500 to-green-600 rounded-xl flex items-center justify-center">
                      <Zap className="w-6 h-6 text-white" />
                    </div>
                    <div className="ml-4">
                      <h3 className="text-xl font-bold text-gray-900">AI Analysis</h3>
                      <p className="text-green-600 font-medium">CNN + LSTM</p>
                    </div>
                  </div>
                  <p className="text-gray-600 leading-relaxed">
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
              <Card hover gradient className="h-full">
                <div className="p-8">
                  <div className="flex items-center mb-6">
                    <div className="w-12 h-12 bg-gradient-to-r from-red-500 to-red-600 rounded-xl flex items-center justify-center">
                      <Target className="w-6 h-6 text-white" />
                    </div>
                    <div className="ml-4">
                      <h3 className="text-xl font-bold text-gray-900">Instant Results</h3>
                      <p className="text-red-600 font-medium">Real-time</p>
                    </div>
                  </div>
                  <p className="text-gray-600 leading-relaxed">
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
            <h3 className="text-2xl font-bold text-gray-900 mb-8">Supported Pitch Types</h3>
            <div className="flex flex-wrap justify-center gap-3">
              {['Fastball', 'Curveball', 'Slider', 'Changeup', 'Knuckleball', 'Cutter'].map((pitch, index) => (
                <motion.div
                  key={pitch}
                  className="px-4 py-2 bg-gradient-to-r from-blue-100 to-blue-200 text-blue-800 rounded-full font-medium"
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
