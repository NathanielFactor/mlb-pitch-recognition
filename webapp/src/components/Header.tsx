'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { motion } from 'framer-motion';
import { Menu, X, Activity } from 'lucide-react';
import { useState } from 'react';
import { BaseballIcon } from './ui';

export default function Header() {
  const pathname = usePathname();
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  const isActive = (path: string) => pathname === path;

  const navItems = [
    { href: '/', label: 'Home' },
    { href: '/upload', label: 'Upload' },
    { href: '/history', label: 'History' }
  ];

  return (
    <header className="bg-black/80 backdrop-blur-md shadow-lg border-b border-white/20">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">
          {/* Logo */}
          <div className="flex items-center space-x-3">
            <motion.div
              whileHover={{ scale: 1.1, rotate: 180 }}
              transition={{ duration: 0.3 }}
            >
              <BaseballIcon size={32} className="text-white" />
            </motion.div>
            <Link href="/" className="flex items-center space-x-2">
              <span className="text-xl font-bold text-white">
                MLB Pitch Recognition
              </span>
              <Activity className="w-5 h-5 text-white/70" />
            </Link>
          </div>

          {/* Desktop Navigation */}
          <nav className="hidden md:flex items-center space-x-1">
            {navItems.map((item) => (
              <Link key={item.href} href={item.href}>
                <motion.div
                  className={`px-4 py-2 rounded-lg font-medium text-sm transition-all duration-200 ${
                    isActive(item.href)
                      ? 'bg-white/25 text-white shadow-lg'
                      : 'text-white/70 hover:text-white hover:bg-white/15'
                  }`}
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                >
                  {item.label}
                  {isActive(item.href) && (
                    <motion.div
                      className="h-0.5 bg-white mt-1 rounded-full"
                      layoutId="activeTab"
                      initial={false}
                    />
                  )}
                </motion.div>
              </Link>
            ))}
          </nav>

          {/* Mobile menu button */}
          <div className="md:hidden">
            <motion.button
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
              className="text-white hover:text-white/70 p-2"
              whileTap={{ scale: 0.95 }}
            >
              {mobileMenuOpen ? <X size={24} /> : <Menu size={24} />}
            </motion.button>
          </div>
        </div>

        {/* Mobile Navigation */}
        {mobileMenuOpen && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="md:hidden border-t border-white/20 mt-2 pt-4 pb-4"
          >
            <nav className="flex flex-col space-y-2">
              {navItems.map((item) => (
                <Link
                  key={item.href}
                  href={item.href}
                  onClick={() => setMobileMenuOpen(false)}
                >
                  <motion.div
                    className={`px-4 py-3 rounded-lg font-medium transition-all duration-200 ${
                      isActive(item.href)
                        ? 'bg-white/25 text-white'
                        : 'text-white/70 hover:text-white hover:bg-white/15'
                    }`}
                    whileTap={{ scale: 0.98 }}
                  >
                    {item.label}
                  </motion.div>
                </Link>
              ))}
            </nav>
          </motion.div>
        )}
      </div>
    </header>
  );
}