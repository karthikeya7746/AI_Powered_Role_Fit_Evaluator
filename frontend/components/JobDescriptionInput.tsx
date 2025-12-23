'use client';

import { useState } from 'react';

interface JobDescriptionInputProps {
  onAnalyze: (jobDescription: string) => void;
  disabled?: boolean;
}

export default function JobDescriptionInput({ onAnalyze, disabled }: JobDescriptionInputProps) {
  const [jobDescription, setJobDescription] = useState('');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (jobDescription.trim()) {
      onAnalyze(jobDescription);
    }
  };

  return (
    <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
      <h2 className="text-xl font-semibold text-gray-900 mb-4">Job Description</h2>
      
      <form onSubmit={handleSubmit}>
        <textarea
          value={jobDescription}
          onChange={(e) => setJobDescription(e.target.value)}
          disabled={disabled}
          placeholder="Paste the job description here..."
          className="w-full h-64 p-4 border border-gray-300 rounded-lg focus:ring-2 focus:ring-gray-900 focus:border-gray-900 resize-none disabled:opacity-50 disabled:cursor-not-allowed text-gray-900 placeholder-gray-400"
        />
        
        <button
          type="submit"
          disabled={disabled || !jobDescription.trim()}
          className="mt-4 w-full bg-gray-900 text-white py-3 px-6 rounded-lg font-medium hover:bg-gray-800 transition-colors disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:bg-gray-900"
        >
          Analyze Match
        </button>
      </form>
    </div>
  );
}


