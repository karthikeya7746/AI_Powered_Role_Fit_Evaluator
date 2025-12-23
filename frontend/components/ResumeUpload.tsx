'use client';

import { useState, useRef } from 'react';

interface ResumeUploadProps {
  onUploaded: (resumeId: string) => void;
  disabled?: boolean;
}

export default function ResumeUpload({ onUploaded, disabled }: ResumeUploadProps) {
  const [uploading, setUploading] = useState(false);
  const [uploadedFile, setUploadedFile] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    if (!file.name.endsWith('.pdf')) {
      alert('Please upload a PDF file');
      return;
    }

    setUploading(true);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}/api/upload-resume`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        let errorMessage = 'Upload failed';
        try {
          const errorData = await response.json();
          if (Array.isArray(errorData.detail)) {
            errorMessage = errorData.detail.map((err: { msg?: string; message?: string }) => err.msg || err.message || JSON.stringify(err)).join(', ');
          } else if (errorData.detail) {
            errorMessage = typeof errorData.detail === 'string' ? errorData.detail : JSON.stringify(errorData.detail);
          } else if (errorData.message) {
            errorMessage = errorData.message;
          } else {
            errorMessage = JSON.stringify(errorData);
          }
        } catch {
          errorMessage = `HTTP ${response.status}: ${response.statusText}`;
        }
        throw new Error(errorMessage);
      }

      const data = await response.json();
      setUploadedFile(file.name);
      onUploaded(data.resume_id);
    } catch (error) {
      let errorMessage = 'Upload failed';
      if (error instanceof Error) {
        errorMessage = error.message;
      } else if (typeof error === 'string') {
        errorMessage = error;
      } else if (error && typeof error === 'object') {
        const errorObj = error as { message?: string; detail?: string };
        errorMessage = errorObj.message || errorObj.detail || JSON.stringify(error);
      }
      alert(errorMessage);
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
      <h2 className="text-xl font-semibold text-gray-900 mb-4">Upload Resume</h2>
      
      <div className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors ${
        disabled || uploading 
          ? 'border-gray-200 bg-gray-50' 
          : uploadedFile
          ? 'border-green-300 bg-green-50'
          : 'border-gray-300 bg-white hover:border-gray-400'
      }`}>
        <input
          ref={fileInputRef}
          type="file"
          accept=".pdf"
          onChange={handleFileChange}
          disabled={disabled || uploading}
          className="hidden"
          id="resume-upload"
        />
        
        <label
          htmlFor="resume-upload"
          className={`flex flex-col items-center ${
            disabled || uploading ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'
          }`}
        >
          {uploading ? (
            <>
              <div className="w-12 h-12 border-3 border-gray-200 border-t-gray-700 rounded-full animate-spin mb-4"></div>
              <p className="text-gray-700">Uploading...</p>
            </>
          ) : uploadedFile ? (
            <>
              <div className="w-12 h-12 bg-green-500 rounded-full flex items-center justify-center mb-4">
                <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M5 13l4 4L19 7" />
                </svg>
              </div>
              <p className="text-gray-900 font-medium mb-1">{uploadedFile}</p>
              <p className="text-sm text-gray-500">Click to upload a different file</p>
            </>
          ) : (
            <>
              <svg
                className="w-12 h-12 text-gray-400 mb-4"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={1.5}
                  d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
                />
              </svg>
              <p className="text-gray-700 font-medium mb-1">
                Click to upload or drag and drop
              </p>
              <p className="text-sm text-gray-500">PDF files only</p>
            </>
          )}
        </label>
      </div>
    </div>
  );
}


