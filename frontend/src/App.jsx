import { useState, useEffect } from 'react'
import Navbar from './components/Navbar'
import Hero from './components/Hero'
import UploadCard from './components/UploadCard'
import LoadingState from './components/LoadingState'
import ErrorState from './components/ErrorState'
import ExperimentHistoryModal from './components/ExperimentHistoryModal'
import BasePaperComparisonModal from './components/BasePaperComparisonModal'
import Home from './pages/Home'
import { uploadAndProfile, fetchSavedRuns, fetchRunById, deleteRunById } from './services/api'
import './App.css'

export default function App() {
  const [status, setStatus] = useState('idle') // 'idle' | 'uploading' | 'profiled' | 'error'
  const [profile, setProfile] = useState(null)
  const [errorMsg, setErrorMsg] = useState('')
  const [uploadingFilename, setUploadingFilename] = useState('')
  const [currentPhase, setCurrentPhase] = useState(1)

  // Experiment History & Persistence State
  const [runs, setRuns] = useState([])
  const [isHistoryOpen, setIsHistoryOpen] = useState(false)
  const [isBenchmarkOpen, setIsBenchmarkOpen] = useState(false)
  const [currentRunId, setCurrentRunId] = useState(null)
  const [initialRunState, setInitialRunState] = useState(null)

  // Load saved runs list from backend on mount
  const loadRuns = async () => {
    try {
      const saved = await fetchSavedRuns()
      setRuns(saved)
    } catch (err) {
      console.warn('Could not load experiment history:', err)
    }
  }

  useEffect(() => {
    loadRuns()
  }, [])

  const handleFileSelected = async (file) => {
    setUploadingFilename(file.name)
    setStatus('uploading')
    setErrorMsg('')
    setCurrentPhase(1)
    setCurrentRunId(null)
    setInitialRunState(null)

    try {
      const data = await uploadAndProfile(file)
      setProfile(data)
      setStatus('profiled')
      setCurrentPhase(1)
    } catch (err) {
      setErrorMsg(err.message || 'An unexpected error occurred while processing the CSV file.')
      setStatus('error')
    }
  }

  // Requirement 4: New Dataset resets only the active workspace, not saved experiment history
  const handleReset = () => {
    setStatus('idle')
    setProfile(null)
    setErrorMsg('')
    setUploadingFilename('')
    setCurrentPhase(1)
    setCurrentRunId(null)
    setInitialRunState(null)
  }

  // Requirement 5: Restores saved run results without re-running computations
  const handleOpenRun = async (runId) => {
    try {
      const runData = await fetchRunById(runId)
      if (!runData || !runData.profile) {
        throw new Error('Experiment run contains no profile data.')
      }

      setCurrentRunId(runData.run_id)
      setProfile(runData.profile)
      setInitialRunState(runData)
      setCurrentPhase(runData.active_phase || 1)
      setStatus('profiled')
    } catch (err) {
      setErrorMsg(err.message || 'Failed to open saved experiment run.')
      setStatus('error')
    }
  }

  const handleDeleteRun = async (runId) => {
    await deleteRunById(runId)
    if (currentRunId === runId) {
      handleReset()
    }
    await loadRuns()
  }

  const handleRunUpdated = (summary) => {
    if (summary?.run_id && summary.run_id !== currentRunId) {
      setCurrentRunId(summary.run_id)
    }
    loadRuns()
  }

  return (
    <div className="app-container">
      <Navbar
        onUploadClick={handleReset}
        hasData={status === 'profiled'}
        currentPhase={currentPhase}
        runCount={runs.length}
        onOpenHistory={() => {
          loadRuns()
          setIsHistoryOpen(true)
        }}
        onOpenBenchmark={() => setIsBenchmarkOpen(true)}
      />

      <main className="page-content">
        {status === 'idle' && (
          <div className="fade-in">
            <Hero />
            <UploadCard onFileSelected={handleFileSelected} />
          </div>
        )}

        {status === 'uploading' && <LoadingState filename={uploadingFilename} />}

        {status === 'error' && <ErrorState message={errorMsg} onRetry={handleReset} />}

        {status === 'profiled' && profile && (
          <Home
            key={currentRunId || profile?.dataset_id || 'active-run'}
            profile={profile}
            onNewUpload={handleReset}
            currentPhase={currentPhase}
            onPhaseChange={setCurrentPhase}
            currentRunId={currentRunId}
            onRunUpdated={handleRunUpdated}
            initialRunState={initialRunState}
          />
        )}
      </main>

      {/* Experiment Run History Modal / Drawer */}
      <ExperimentHistoryModal
        isOpen={isHistoryOpen}
        onClose={() => setIsHistoryOpen(false)}
        runs={runs}
        activeRunId={currentRunId}
        activeRunData={initialRunState}
        onOpenRun={handleOpenRun}
        onDeleteRun={handleDeleteRun}
        onNewUpload={handleReset}
      />

      {/* AutoMind Table 3 Classification Benchmark Modal */}
      <BasePaperComparisonModal
        isOpen={isBenchmarkOpen}
        onClose={() => setIsBenchmarkOpen(false)}
      />

      <footer className="app-footer">
        <div className="app-footer-inner">
          <span>DataInvest AI · Automated Statistical Pipeline (Phases 1–7)</span>
          <span className="footer-meta">Powered by FastAPI, SciPy, Pandas & React</span>
        </div>
      </footer>
    </div>
  )
}


