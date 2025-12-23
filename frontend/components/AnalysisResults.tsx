'use client';

import { AnalysisResult } from '@/types';
import { useState } from 'react';

interface AnalysisResultsProps {
  result: AnalysisResult;
}

export default function AnalysisResults({ result }: AnalysisResultsProps) {
  const [activeTab, setActiveTab] = useState<'overview' | 'gaps' | 'strengths' | 'evidence' | 'tailored'>('overview');

  const getScoreColor = (score: number) => {
    if (score >= 80) return 'bg-green-100 text-green-700 border-green-300';
    if (score >= 60) return 'bg-yellow-100 text-yellow-700 border-yellow-300';
    return 'bg-red-100 text-red-700 border-red-300';
  };

  const exportToMarkdown = () => {
    const markdown = `# Resume-Job Match Analysis

## Fit Score: ${result.fit_score}/100

### Strengths
${result.strengths.map(s => `- ${s}`).join('\n')}

### Gaps
${result.gaps.map(g => `- **${g.requirement}** (${g.status}): ${g.evidence || 'N/A'}`).join('\n')}

### Tailored Bullets
${result.tailored_bullets.map(b => `- ${b}`).join('\n')}

### Cover Letter Snippet
${result.cover_letter_snippet || 'N/A'}
`;

    const blob = new Blob([markdown], { type: 'text/markdown' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'resume-analysis.md';
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-8 mt-8">
      <div className="flex justify-between items-center mb-8 pb-6 border-b border-gray-200">
        <h2 className="text-2xl font-semibold text-gray-900">Analysis Results</h2>
        <button
          onClick={exportToMarkdown}
          className="px-4 py-2 bg-gray-900 text-white rounded-lg font-medium hover:bg-gray-800 transition-colors flex items-center space-x-2"
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
          </svg>
          <span>Export to Markdown</span>
        </button>
      </div>

      {/* Fit Score */}
      <div className="mb-8 flex items-center justify-center">
        <div className={`inline-flex items-center justify-center w-32 h-32 rounded-full border-4 ${getScoreColor(result.fit_score)}`}>
          <div className="text-center">
            <div className="text-4xl font-bold">{result.fit_score}</div>
            <div className="text-sm font-medium">/ 100</div>
          </div>
        </div>
      </div>

      {/* Tabs */}
      <div className="border-b border-gray-200 mb-6">
        <nav className="flex space-x-1">
          {([
            { id: 'overview' as const, label: 'Overview' },
            { id: 'strengths' as const, label: 'Strengths' },
            { id: 'gaps' as const, label: 'Gaps' },
            { id: 'evidence' as const, label: 'Evidence' },
            { id: 'tailored' as const, label: 'Tailored Content' },
          ] as const).map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`py-3 px-6 border-b-2 font-medium text-sm transition-colors ${
                activeTab === tab.id
                  ? 'border-gray-900 text-gray-900'
                  : 'border-transparent text-gray-600 hover:text-gray-900 hover:border-gray-300'
              }`}
            >
              {tab.label}
            </button>
          ))}
        </nav>
      </div>

      {/* Tab Content */}
      <div className="min-h-[400px]">
        {activeTab === 'overview' && (
          <div className="space-y-6">
            <div>
              <h3 className="font-semibold text-lg mb-3 text-gray-900">Fit Score: {result.fit_score}/100</h3>
              <div className="w-full bg-gray-200 rounded-full h-3 mb-4">
                <div
                  className={`h-3 rounded-full transition-all duration-500 ${
                    result.fit_score >= 80
                      ? 'bg-green-600'
                      : result.fit_score >= 60
                      ? 'bg-yellow-500'
                      : 'bg-red-500'
                  }`}
                  style={{ width: `${result.fit_score}%` }}
                />
              </div>
            </div>
            {result.cover_letter_snippet && (
              <div className="border-l-4 border-gray-300 pl-6">
                <h3 className="font-semibold text-lg mb-3 text-gray-900">Cover Letter Snippet</h3>
                <p className="text-gray-700 leading-relaxed">{result.cover_letter_snippet}</p>
              </div>
            )}
          </div>
        )}

        {activeTab === 'strengths' && (
          <div>
            <h3 className="font-semibold text-lg mb-4 text-gray-900">Key Strengths</h3>
            <ul className="space-y-3">
              {result.strengths.map((strength, idx) => (
                <li key={idx} className="flex items-start">
                  <svg className="w-5 h-5 text-green-600 mr-3 mt-0.5 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                  </svg>
                  <span className="text-gray-700">{strength}</span>
                </li>
              ))}
            </ul>
          </div>
        )}

        {activeTab === 'gaps' && (
          <div>
            <h3 className="font-semibold text-lg mb-4 text-gray-900">Gaps & Missing Requirements</h3>
            <div className="space-y-4">
              {result.gaps.map((gap, idx) => (
                <div key={idx} className="border-l-4 border-orange-400 pl-4 py-3">
                  <div className="font-medium text-gray-900 mb-1">{gap.requirement}</div>
                  <div className="text-sm text-gray-600 mb-2">
                    Status: <span className="font-medium">{gap.status}</span>
                  </div>
                  {gap.evidence && (
                    <div className="text-sm text-gray-700 mt-2">{gap.evidence}</div>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}

        {activeTab === 'evidence' && (
          <div>
            <h3 className="font-semibold text-lg mb-4 text-gray-900">Supporting Evidence</h3>
            <div className="space-y-4">
              {result.evidence.map((item, idx) => (
                <div key={idx} className="border border-gray-200 rounded-lg p-5">
                  <div className="font-medium text-gray-900 mb-3">{item.requirement}</div>
                  <div className="text-gray-700 bg-gray-50 p-4 rounded mb-3 text-sm">
                    {item.resume_evidence}
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-xs text-gray-600">Relevance:</span>
                    <span className="text-sm font-medium text-gray-900">{(item.relevance_score * 100).toFixed(1)}%</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {activeTab === 'tailored' && (
          <div>
            <h3 className="font-semibold text-lg mb-4 text-gray-900">Tailored Resume Bullets</h3>
            <ul className="space-y-3 mb-6">
              {result.tailored_bullets.map((bullet, idx) => (
                <li key={idx} className="flex items-start">
                  <span className="text-gray-900 mr-3 font-medium">{idx + 1}.</span>
                  <span className="text-gray-700">{bullet}</span>
                </li>
              ))}
            </ul>
            {result.cover_letter_snippet && (
              <div className="border-l-4 border-gray-300 pl-6 pt-4">
                <h3 className="font-semibold text-lg mb-3 text-gray-900">Cover Letter Snippet</h3>
                <p className="text-gray-700 leading-relaxed">{result.cover_letter_snippet}</p>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}


