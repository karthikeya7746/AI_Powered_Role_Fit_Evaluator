'use client';

import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Sparkles, Download, ChevronDown, Loader2, FileText, Zap } from 'lucide-react';
import axios from 'axios';
import FileUpload from '@/components/FileUpload';
import CircularGauge from '@/components/CircularGauge';

interface EvaluationResult {
  fit_score: number;
  strengths: string[];
  gaps: string[];
  evidence: string[];
  cover_letter?: string;
}

// Container animation variants for stagger effect
const containerVariants = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: {
      staggerChildren: 0.1,
      delayChildren: 0.2,
    },
  },
};

const itemVariants = {
  hidden: { opacity: 0, y: 20 },
  visible: {
    opacity: 1,
    y: 0,
    transition: {
      duration: 0.5,
      ease: 'easeOut',
    },
  },
};

export default function Home() {
  const [resumeFile, setResumeFile] = useState<File | null>(null);
  const [jobDescription, setJobDescription] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<EvaluationResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [evidenceExpanded, setEvidenceExpanded] = useState(false);
  const [coverLetterExpanded, setCoverLetterExpanded] = useState(false);

  // Use external backend if set (e.g. Render); otherwise same-origin /api (Vercel-only)
  const API_BASE = process.env.NEXT_PUBLIC_API_URL;
  const evaluateUrl = API_BASE ? `${API_BASE.replace(/\/$/, '')}/evaluate` : '/api/evaluate';

  const handleFileSelect = (file: File) => {
    setResumeFile(file);
    setError(null);
    setResult(null);
  };

  const handleEvaluate = async () => {
    if (!resumeFile) {
      setError('Please upload a resume file');
      return;
    }

    if (!jobDescription.trim()) {
      setError('Please enter a job description');
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const formData = new FormData();
      formData.append('resume', resumeFile);
      formData.append('job_description', jobDescription);

      const response = await axios.post<EvaluationResult>(
        evaluateUrl,
        formData,
        {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
        }
      );

      setResult(response.data);
      setEvidenceExpanded(false);
      setCoverLetterExpanded(false);
    } catch (err) {
      if (axios.isAxiosError(err)) {
        setError(err.response?.data?.detail || err.message || 'Evaluation failed');
      } else {
        setError('An unexpected error occurred');
      }
    } finally {
      setLoading(false);
    }
  };

  const handleDownloadReport = () => {
    if (!result) return;

    const report = `AI-Powered Role Fit Evaluation Report
========================================

Fit Score: ${result.fit_score}%

TOP STRENGTHS
-------------
${result.strengths.map((s, i) => `${i + 1}. ${s}`).join('\n')}

GAPS & MISSING REQUIREMENTS
----------------------------
${result.gaps.map((g, i) => `${i + 1}. ${g}`).join('\n')}

SUPPORTING EVIDENCE
-------------------
${result.evidence.map((e, i) => `${i + 1}. ${e}`).join('\n')}

${result.cover_letter ? `\nGENERATED COVER LETTER\n-------------------\n${result.cover_letter}\n` : ''}

Generated on: ${new Date().toLocaleString()}
`;

    const blob = new Blob([report], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'role-fit-evaluation-report.txt';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-indigo-50 via-blue-50 to-indigo-50">
      {/* Subtle background decoration */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-0 right-0 w-96 h-96 bg-blue-200/10 rounded-full blur-3xl"></div>
        <div className="absolute bottom-0 left-0 w-96 h-96 bg-purple-200/10 rounded-full blur-3xl"></div>
      </div>

      <div className="relative z-10 container mx-auto px-4 py-8 md:py-12 max-w-6xl min-h-screen flex flex-col">
        {/* Header Section */}
        <motion.header
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          className="text-center mb-8 md:mb-12"
        >
          <motion.h1
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 0.1, duration: 0.5 }}
            className="text-4xl md:text-5xl lg:text-6xl font-bold text-gray-800 mb-4"
          >
            Role Fit Evaluator
          </motion.h1>
          <motion.p
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.3 }}
            className="text-lg md:text-xl text-gray-600 max-w-2xl mx-auto"
          >
            Upload your resume and job description to get an instant AI-powered fit score analysis
          </motion.p>
        </motion.header>

        {/* Main Input Card */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.2 }}
          className="bg-white/80 backdrop-blur-lg rounded-2xl shadow-xl border border-white/20 p-6 md:p-8 mb-8"
        >
          {/* Error Message */}
          <AnimatePresence>
            {error && (
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: 'auto' }}
                exit={{ opacity: 0, height: 0 }}
                className="mb-6 p-4 bg-red-50 border-l-4 border-red-500 rounded-lg"
              >
                <p className="text-red-800 text-sm font-medium">{error}</p>
              </motion.div>
            )}
          </AnimatePresence>

          <div className="space-y-6">
            {/* Resume Upload Section */}
            <div>
              <label className="block text-sm font-semibold text-gray-800 mb-3 flex items-center space-x-2">
                <FileText className="w-4 h-4" />
                <span>Upload Resume</span>
              </label>
              <FileUpload
                onFileSelect={handleFileSelect}
                disabled={loading}
                acceptedTypes={['.pdf', '.txt', '.docx', '.doc']}
              />
            </div>

            {/* Job Description Section */}
            <div>
              <label className="block text-sm font-semibold text-gray-800 mb-3 flex items-center space-x-2">
                <Zap className="w-4 h-4" />
                <span>Job Description</span>
              </label>
              <textarea
                value={jobDescription}
                onChange={(e) => setJobDescription(e.target.value)}
                disabled={loading}
                placeholder="Paste or type the job description here..."
                className="w-full h-48 p-4 border border-gray-300 rounded-xl focus:ring-2 focus:ring-blue-600 focus:border-blue-600 resize-none disabled:opacity-50 disabled:cursor-not-allowed bg-white/70 backdrop-blur-sm text-gray-800 placeholder-gray-400 transition-all"
              />
            </div>

            {/* Evaluate Button */}
            <motion.button
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              onClick={handleEvaluate}
              disabled={loading || !resumeFile || !jobDescription.trim()}
              className="w-full py-4 px-6 bg-gradient-to-r from-blue-600 to-purple-600 text-white font-semibold rounded-xl shadow-lg hover:shadow-xl transition-all disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:scale-100 flex items-center justify-center space-x-2"
            >
              {loading ? (
                <>
                  <Loader2 className="w-5 h-5 animate-spin" />
                  <span>Evaluating...</span>
                </>
              ) : (
                <>
                  <Sparkles className="w-5 h-5" />
                  <span>Evaluate Resume</span>
                </>
              )}
            </motion.button>
          </div>
        </motion.div>

        {/* Loading State */}
        <AnimatePresence>
          {loading && (
            <motion.div
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.9 }}
              className="bg-white/80 backdrop-blur-lg rounded-2xl shadow-xl border border-white/20 p-8 md:p-12 text-center mb-8"
            >
              <motion.div
                animate={{ rotate: 360 }}
                transition={{ duration: 1, repeat: Infinity, ease: 'linear' }}
                className="inline-block mb-6"
              >
                <Loader2 className="w-16 h-16 text-blue-600" />
              </motion.div>
              <p className="text-gray-800 font-semibold text-lg mb-2">Analyzing your resume...</p>
              <p className="text-sm text-gray-600">This may take a few moments</p>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Results Section */}
        <AnimatePresence>
          {result && !loading && (
            <motion.div
              variants={containerVariants}
              initial="hidden"
              animate="visible"
              exit="hidden"
              className="space-y-6 mb-8"
            >
              {/* Fit Score Card with Circular Gauge */}
              <motion.div
                variants={itemVariants}
                className="bg-white/80 backdrop-blur-lg rounded-2xl shadow-xl border border-white/20 p-8 md:p-12 text-center"
              >
                <h2 className="text-2xl font-bold text-gray-800 mb-8">Your Fit Score</h2>
                <div className="flex justify-center mb-8">
                  <CircularGauge score={result.fit_score} size={220} strokeWidth={18} />
                </div>
                <div className="w-full max-w-md mx-auto bg-gray-200 rounded-full h-2 overflow-hidden">
                  <motion.div
                    initial={{ width: 0 }}
                    animate={{ width: `${result.fit_score}%` }}
                    transition={{ delay: 0.5, duration: 1.2, ease: 'easeOut' }}
                    className={`h-full rounded-full ${
                      result.fit_score >= 80
                        ? 'bg-gradient-to-r from-green-500 to-emerald-600'
                        : result.fit_score >= 60
                        ? 'bg-gradient-to-r from-yellow-500 to-orange-500'
                        : 'bg-gradient-to-r from-red-500 to-pink-600'
                    }`}
                  />
                </div>
              </motion.div>

              {/* Strengths and Gaps Grid */}
              <div className="grid md:grid-cols-2 gap-6">
                {/* Strengths Card */}
                <motion.div
                  variants={itemVariants}
                  whileHover={{ scale: 1.02 }}
                  className="bg-white/80 backdrop-blur-lg rounded-2xl shadow-xl border border-white/20 p-6"
                >
                  <div className="flex items-center space-x-3 mb-6">
                    <div className="p-2.5 bg-green-100 rounded-lg">
                      <Sparkles className="w-5 h-5 text-green-700" />
                    </div>
                    <h3 className="text-xl font-bold text-gray-800">Strengths</h3>
                  </div>
                  <div className="flex flex-wrap gap-2.5">
                    {result.strengths.map((strength, idx) => (
                      <motion.span
                        key={idx}
                        initial={{ opacity: 0, scale: 0.8 }}
                        animate={{ opacity: 1, scale: 1 }}
                        transition={{ delay: 0.6 + idx * 0.05 }}
                        whileHover={{ scale: 1.05 }}
                        className="px-4 py-2 bg-green-100 text-green-700 rounded-full text-sm font-semibold shadow-sm"
                      >
                        {strength}
                      </motion.span>
                    ))}
                  </div>
                </motion.div>

                {/* Gaps Card */}
                <motion.div
                  variants={itemVariants}
                  whileHover={{ scale: 1.02 }}
                  className="bg-white/80 backdrop-blur-lg rounded-2xl shadow-xl border border-white/20 p-6"
                >
                  <div className="flex items-center space-x-3 mb-6">
                    <div className="p-2.5 bg-yellow-100 rounded-lg">
                      <FileText className="w-5 h-5 text-yellow-700" />
                    </div>
                    <h3 className="text-xl font-bold text-gray-800">Gaps</h3>
                  </div>
                  <div className="flex flex-wrap gap-2.5">
                    {result.gaps.map((gap, idx) => (
                      <motion.span
                        key={idx}
                        initial={{ opacity: 0, scale: 0.8 }}
                        animate={{ opacity: 1, scale: 1 }}
                        transition={{ delay: 0.7 + idx * 0.05 }}
                        whileHover={{ scale: 1.05 }}
                        className="px-4 py-2 bg-yellow-100 text-yellow-700 rounded-full text-sm font-semibold shadow-sm"
                      >
                        {gap}
                      </motion.span>
                    ))}
                  </div>
                </motion.div>
              </div>

              {/* Evidence Panel */}
              <motion.div
                variants={itemVariants}
                whileHover={{ scale: 1.01 }}
                className="bg-white/80 backdrop-blur-lg rounded-2xl shadow-xl border border-white/20 overflow-hidden"
              >
                <button
                  onClick={() => setEvidenceExpanded(!evidenceExpanded)}
                  className="w-full p-6 flex items-center justify-between hover:bg-gray-50/50 transition-colors"
                >
                  <div className="flex items-center space-x-3">
                    <div className="p-2.5 bg-blue-100 rounded-lg">
                      <FileText className="w-5 h-5 text-blue-700" />
                    </div>
                    <h3 className="text-xl font-bold text-gray-800">
                      Evidence / Justification
                    </h3>
                  </div>
                  <motion.div
                    animate={{ rotate: evidenceExpanded ? 180 : 0 }}
                    transition={{ duration: 0.3 }}
                  >
                    <ChevronDown className="w-5 h-5 text-gray-500" />
                  </motion.div>
                </button>
                <AnimatePresence>
                  {evidenceExpanded && (
                    <motion.div
                      initial={{ height: 0, opacity: 0 }}
                      animate={{ height: 'auto', opacity: 1 }}
                      exit={{ height: 0, opacity: 0 }}
                      transition={{ duration: 0.3 }}
                      className="overflow-hidden"
                    >
                      <div className="px-6 pb-6 space-y-3">
                        {result.evidence.map((item, idx) => (
                          <motion.div
                            key={idx}
                            initial={{ opacity: 0, x: -10 }}
                            animate={{ opacity: 1, x: 0 }}
                            transition={{ delay: idx * 0.05 }}
                            className="p-4 bg-gray-50 rounded-lg border-l-4 border-blue-600"
                          >
                            <p className="text-gray-800 text-sm leading-relaxed">{item}</p>
                          </motion.div>
                        ))}
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>
              </motion.div>

              {/* Cover Letter Panel */}
              {result.cover_letter && (
                <motion.div
                  variants={itemVariants}
                  whileHover={{ scale: 1.01 }}
                  className="bg-white/80 backdrop-blur-lg rounded-2xl shadow-xl border border-white/20 overflow-hidden"
                >
                  <button
                    onClick={() => setCoverLetterExpanded(!coverLetterExpanded)}
                    className="w-full p-6 flex items-center justify-between hover:bg-gray-50/50 transition-colors"
                  >
                    <div className="flex items-center space-x-3">
                      <div className="p-2.5 bg-purple-100 rounded-lg">
                        <FileText className="w-5 h-5 text-purple-700" />
                      </div>
                      <h3 className="text-xl font-bold text-gray-800">
                        Generated Cover Letter
                      </h3>
                    </div>
                    <motion.div
                      animate={{ rotate: coverLetterExpanded ? 180 : 0 }}
                      transition={{ duration: 0.3 }}
                    >
                      <ChevronDown className="w-5 h-5 text-gray-500" />
                    </motion.div>
                  </button>
                  <AnimatePresence>
                    {coverLetterExpanded && (
                      <motion.div
                        initial={{ height: 0, opacity: 0 }}
                        animate={{ height: 'auto', opacity: 1 }}
                        exit={{ height: 0, opacity: 0 }}
                        transition={{ duration: 0.3 }}
                        className="overflow-hidden"
                      >
                        <div className="px-6 pb-6">
                          <div className="p-6 bg-gradient-to-br from-purple-50 to-blue-50 rounded-lg border border-purple-200">
                            <div className="flex justify-end mb-4">
                              <motion.button
                                whileHover={{ scale: 1.05 }}
                                whileTap={{ scale: 0.95 }}
                                onClick={() => {
                                  const blob = new Blob([result.cover_letter || ''], { type: 'text/plain' });
                                  const url = URL.createObjectURL(blob);
                                  const a = document.createElement('a');
                                  a.href = url;
                                  a.download = 'cover-letter.txt';
                                  document.body.appendChild(a);
                                  a.click();
                                  document.body.removeChild(a);
                                  URL.revokeObjectURL(url);
                                }}
                                className="px-4 py-2 bg-purple-600 text-white text-sm font-semibold rounded-lg hover:bg-purple-700 transition-colors flex items-center space-x-2"
                              >
                                <Download className="w-4 h-4" />
                                <span>Download Cover Letter</span>
                              </motion.button>
                            </div>
                            <div className="prose prose-sm max-w-none">
                              <p className="text-gray-800 leading-relaxed whitespace-pre-wrap font-serif">
                                {result.cover_letter}
                              </p>
                            </div>
                          </div>
                        </div>
                      </motion.div>
                    )}
                  </AnimatePresence>
                </motion.div>
              )}

              {/* Download Report Button */}
              <motion.div
                variants={itemVariants}
                className="flex justify-center pt-4"
              >
                <motion.button
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  onClick={handleDownloadReport}
                  className="px-8 py-3.5 bg-gradient-to-r from-blue-600 to-purple-600 text-white font-semibold rounded-xl shadow-lg hover:shadow-xl transition-all flex items-center space-x-2"
                >
                  <Download className="w-5 h-5" />
                  <span>Download Report</span>
                </motion.button>
              </motion.div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Footer */}
        <motion.footer
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.8 }}
          className="mt-auto pt-8 pb-4 text-center"
        >
          <p className="text-gray-600 text-sm">
            Find your perfect job match — powered by AI.
          </p>
        </motion.footer>
      </div>
    </div>
  );
}
