// components/UploadCard.jsx
import { useRef, useState, useCallback } from 'react'
import './UploadCard.css'

export default function UploadCard({ onFileSelected }) {
  const inputRef = useRef(null)
  const [dragOver, setDragOver] = useState(false)

  const handleFile = useCallback((file) => {
    if (!file) return
    const name = file.name.toLowerCase()
    if (!name.endsWith('.csv') && !name.endsWith('.txt')) {
      alert('Please upload a CSV or TXT file.')
      return
    }
    onFileSelected(file)
  }, [onFileSelected])

  const onDrop = useCallback((e) => {
    e.preventDefault()
    setDragOver(false)
    const file = e.dataTransfer.files[0]
    handleFile(file)
  }, [handleFile])

  const onDragOver = useCallback((e) => { e.preventDefault(); setDragOver(true) }, [])
  const onDragLeave = useCallback(() => setDragOver(false), [])

  const onInputChange = (e) => {
    const file = e.target.files[0]
    handleFile(file)
    e.target.value = ''
  }

  return (
    <div
      className={`upload-card${dragOver ? ' drag-over' : ''}`}
      onClick={() => inputRef.current?.click()}
      onDrop={onDrop}
      onDragOver={onDragOver}
      onDragLeave={onDragLeave}
      tabIndex={0}
      onKeyDown={(e) => e.key === 'Enter' && inputRef.current?.click()}
      role="button"
      aria-label="Upload CSV or TXT file"
    >
      <div className="upload-icon">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
          <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
          <polyline points="17 8 12 3 7 8"/>
          <line x1="12" y1="3" x2="12" y2="15"/>
        </svg>
      </div>
      <div className="upload-title">Upload Dataset</div>
      <div className="upload-subtitle">Drop your file here or click to browse</div>
      <button
        className="upload-browse-btn"
        onClick={(e) => { e.stopPropagation(); inputRef.current?.click() }}
      >
        Browse File
      </button>
      <div className="upload-hint">CSV or TXT files · Max 200 MB</div>
      <input
        ref={inputRef}
        type="file"
        accept=".csv,.txt"
        className="upload-hidden-input"
        onChange={onInputChange}
      />
    </div>
  )
}
