// services/api.js — DataInvest AI API client

/**
 * Resolves the backend API base URL safely across Vite local dev and production deployments.
 * - In local dev (or when unset): defaults to '/api', using Vite's server.proxy.
 * - When configured (e.g. 'https://datainvest-api.onrender.com' or 'https://.../api'):
 *   strips trailing slashes and guarantees exactly one '/api' prefix for all endpoints.
 */
export function resolveApiBaseUrl(overrideUrl) {
  const envUrl = (overrideUrl !== undefined ? overrideUrl : (typeof import.meta !== 'undefined' && import.meta.env ? import.meta.env.VITE_API_BASE_URL : '')) || ''
  const trimmed = String(envUrl).trim()
  if (!trimmed) {
    return '/api'
  }
  const cleaned = trimmed.replace(/\/+$/, '')
  return cleaned.endsWith('/api') ? cleaned : `${cleaned}/api`
}

const BASE_URL = resolveApiBaseUrl()

/**
 * Upload a CSV file and return the profile JSON (Phase 1).
 * @param {File} file
 * @returns {Promise<Object>} profile data
 * @throws {Error} with user-friendly message
 */
export async function uploadAndProfile(file) {
  const formData = new FormData()
  formData.append('file', file)

  const response = await fetch(`${BASE_URL}/profile`, {
    method: 'POST',
    body: formData,
  })

  const data = await response.json()

  if (!response.ok) {
    throw new Error(data.error || 'Upload failed. Please try again.')
  }

  return data
}

/**
 * Generate an analytical investigation plan from dataset profile and objective (Phase 2).
 * @param {Object} profile - Phase 1 profile object
 * @param {string} objective - User analytical goal
 * @returns {Promise<Object>} PlannerResponse
 * @throws {Error} with user-friendly message
 */
export async function createInvestigationPlan(profile, objective = 'Investigate this dataset.', options = {}) {
  const payload = {
    profile,
    objective: objective || 'Investigate this dataset.',
    dataset_id: profile?.dataset_id || profile?.filename || 'sample.csv',
    use_llm: options?.use_llm !== undefined ? Boolean(options.use_llm) : true,
  }
  if (options && typeof options === 'object') {
    payload.preferences = { ...(options.preferences || {}), use_llm: payload.use_llm }
  }

  const response = await fetch(`${BASE_URL}/plan`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(payload),
  })

  const data = await response.json()

  if (!response.ok) {
    throw new Error(data.error || 'Unable to generate analysis plan. Please check inputs and retry.')
  }

  return data
}

/**
 * Execute an entire planned list of investigation tasks via SciPy tools (Phase 3).
 * @param {string} datasetId - Cached dataset identifier or filename
 * @param {Array} tasks - Planned tasks from PlannerResponse
 * @returns {Promise<Object>} PlanAnalysisResponse
 * @throws {Error} with user-friendly message
 */
export async function executeAnalysisPlan(datasetId, tasks) {
  const response = await fetch(`${BASE_URL}/analyze-plan`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      dataset_id: datasetId || 'sample.csv',
      tasks: tasks,
    }),
  })

  const data = await response.json()

  if (!response.ok) {
    throw new Error(data.error || 'Failed to execute statistical analysis plan.')
  }

  return data
}

/**
 * Execute a single investigation task (Phase 3).
 * @param {string} datasetId - Cached dataset identifier or filename
 * @param {Object} task - Single investigation task
 * @returns {Promise<Object>} AnalysisResult
 * @throws {Error} with user-friendly message
 */
export async function executeSingleTask(datasetId, task) {
  const response = await fetch(`${BASE_URL}/analyze`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      dataset_id: datasetId || 'sample.csv',
      task: task,
    }),
  })

  const data = await response.json()

  if (!response.ok) {
    throw new Error(data.error || 'Failed to execute task.')
  }

  return data
}

/**
 * Run Critic verification on Phase 3 analysis findings (Phase 4).
 * @param {Object} analysisResponse - Phase 3 PlanAnalysisResponse
 * @param {Object} profile - Phase 1 profile object
 * @param {Object} plan - Phase 2 planner response
 * @returns {Promise<Object>} CriticResponse
 * @throws {Error} with user-friendly message
 */
export async function runCriticReview(analysisResponse, profile = null, plan = null) {
  const response = await fetch(`${BASE_URL}/critic`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      analysis_response: analysisResponse,
      dataset_profile: profile,
      investigation_plan: plan,
      dataset_id: analysisResponse?.dataset_id || profile?.dataset_id || 'sample.csv',
    }),
  })

  const data = await response.json()

  if (!response.ok) {
    throw new Error(data.error || 'Failed to execute Critic verification review.')
  }

  return data
}

/**
 * Execute Phase 5 Re-investigation & Verified Findings orchestration.
 * @param {Object} analysisResponse - Phase 3 PlanAnalysisResponse
 * @param {Object} criticResponse - Phase 4 CriticResponse
 * @param {Object} profile - Phase 1 profile object
 * @param {Object} plan - Phase 2 planner response
 * @param {number} maxReinvestigations - Loop guard limit (default: 1)
 * @returns {Promise<Object>} VerificationResponse
 * @throws {Error} with user-friendly message
 */
export async function verifyFindings(
  analysisResponse,
  criticResponse,
  profile = null,
  plan = null,
  maxReinvestigations = 1
) {
  const response = await fetch(`${BASE_URL}/verify-findings`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      analysis_response: analysisResponse,
      critic_response: criticResponse,
      dataset_profile: profile,
      investigation_plan: plan,
      dataset_id:
        analysisResponse?.dataset_id ||
        criticResponse?.dataset_id ||
        profile?.dataset_id ||
        'sample.csv',
      max_reinvestigations: maxReinvestigations,
    }),
  })

  const data = await response.json()

  if (!response.ok) {
    throw new Error(data.error || 'Failed to execute finding verification and re-investigation.')
  }

  return data
}

/**
 * Generate human-readable analytical insights from Phase 5 verified findings (Phase 6).
 * @param {Object} verificationResponse - Phase 5 VerificationResponse
 * @param {Object} profile - Phase 1 profile object
 * @param {Object} plan - Phase 2 planner response
 * @returns {Promise<Object>} InsightResponse
 * @throws {Error} with user-friendly message
 */
export async function generateInsights(verificationResponse, profile = null, plan = null) {
  const response = await fetch(`${BASE_URL}/insights`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      verification_response: verificationResponse,
      dataset_profile: profile,
      investigation_plan: plan,
      dataset_id:
        verificationResponse?.dataset_id ||
        profile?.dataset_id ||
        'sample.csv',
    }),
  })

  const data = await response.json()

  if (!response.ok) {
    throw new Error(data.error || 'Failed to generate analytical insights.')
  }

  return data
}

/**
 * Run Phase 7 System Evaluation across all completed pipeline stages.
 * @param {Object} payload - { profile, plan, analysis, critic, verification, insights }
 * @returns {Promise<Object>} EvaluationResponse
 * @throws {Error} with user-friendly message
 */
export async function evaluatePipeline({
  profile,
  plan = null,
  analysis = null,
  critic = null,
  verification = null,
  insights = null,
}) {
  const response = await fetch(`${BASE_URL}/evaluate`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      profile,
      plan,
      analysis,
      critic,
      verification,
      insights,
    }),
  })

  const data = await response.json()

  if (!response.ok) {
    throw new Error(data.error || 'Failed to evaluate system pipeline.')
  }

  return data
}

/**
 * Fetch all saved experiment runs sorted newest first.
 * @returns {Promise<Array>} List of run summaries
 */
export async function fetchSavedRuns() {
  const response = await fetch(`${BASE_URL}/runs`)
  const data = await response.json()
  if (!response.ok) {
    throw new Error(data.error || 'Failed to load experiment history.')
  }
  return data.runs || []
}

/**
 * Fetch full serialized experiment run payload by runId.
 * @param {string} runId
 * @returns {Promise<Object>} Complete RunPayload
 */
export async function fetchRunById(runId) {
  const response = await fetch(`${BASE_URL}/runs/${encodeURIComponent(runId)}`)
  const data = await response.json()
  if (!response.ok) {
    throw new Error(data.error || `Failed to load experiment run '${runId}'.`)
  }
  return data
}

/**
 * Save or auto-update an experiment run.
 * @param {Object} payload - RunPayload
 * @returns {Promise<Object>} Save response with run_id and summary
 */
export async function saveRun(payload) {
  const response = await fetch(`${BASE_URL}/runs`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(payload),
  })
  const data = await response.json()
  if (!response.ok) {
    throw new Error(data.error || 'Failed to save experiment run.')
  }
  return data
}

/**
 * Delete an experiment run from persistent storage.
 * @param {string} runId
 * @returns {Promise<Object>}
 */
export async function deleteRunById(runId) {
  const response = await fetch(`${BASE_URL}/runs/${encodeURIComponent(runId)}`, {
    method: 'DELETE',
  })
  const data = await response.json()
  if (!response.ok) {
    throw new Error(data.error || `Failed to delete experiment run '${runId}'.`)
  }
  return data
}

/**
 * Execute a scientific paired experiment comparing Baseline vs Proposed architectures.
 * Dispatches POST /api/experiments/paired strictly matching PairedExperimentRequest schema.
 * @param {Object} params
 * @param {string} params.datasetId - Dataset identifier or filename
 * @param {Object} [params.profile] - Phase 1 dataset profile (optional)
 * @param {string} [params.objective] - User investigation goal in natural language
 * @param {number} [params.maxReinvestigations=1] - Maximum re-investigation loops (0-3)
 * @param {boolean} [params.persistRuns=true] - Whether to persist both branch runs to disk
 * @returns {Promise<Object>} PairedExperimentResponse
 */
export async function runPairedExperiment({
  datasetId,
  profile,
  objective = 'Investigate this dataset.',
  maxReinvestigations = 1,
  persistRuns = true,
}) {
  const resolvedDatasetId = datasetId || profile?.dataset_id || profile?.filename || 'sample.csv'
  const payload = {
    dataset_id: resolvedDatasetId,
    objective: objective || 'Investigate this dataset.',
    max_reinvestigations: typeof maxReinvestigations === 'number' ? maxReinvestigations : 1,
    persist_runs: persistRuns !== false,
  }
  if (profile) {
    payload.profile = profile
  }

  const response = await fetch(`${BASE_URL}/experiments/paired`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(payload),
  })

  const data = await response.json()

  if (!response.ok) {
    throw new Error(data.error || 'Failed to execute comparative experiment.')
  }

  return data
}

/**
 * Fetch structured scientific research report for a specific saved run.
 * @param {string} runId
 * @returns {Promise<Object>} ResearchReport
 */
export async function fetchRunReport(runId) {
  const response = await fetch(`${BASE_URL}/runs/${encodeURIComponent(runId)}/report`)
  const data = await response.json()
  if (!response.ok) {
    throw new Error(data.error || `Failed to load report for run '${runId}'.`)
  }
  return data
}

/**
 * Trigger download of Markdown or JSON export for a specific saved run.
 * Uses native browser blob download ensuring strict correspondence to runId.
 * @param {string} runId
 * @param {'markdown' | 'json'} format
 * @returns {Promise<{ success: boolean, filename: string }>}
 */
export async function downloadRunExport(runId, format = 'markdown') {
  if (!runId) {
    throw new Error('A valid runId is required to export.')
  }
  const endpoint = format === 'json' ? 'export/json' : 'export/markdown'
  const response = await fetch(`${BASE_URL}/runs/${encodeURIComponent(runId)}/${endpoint}`)
  if (!response.ok) {
    let errMessage = `Failed to download ${format} export for run '${runId}'.`
    try {
      const errData = await response.json()
      if (errData?.error) errMessage = errData.error
    } catch (_) {}
    if (response.status === 404 && !errMessage.includes('restart')) {
      errMessage += ' (Run record not found on backend storage. On free-tier cloud deployments, dynamically created runs expire upon container restart).'
    }
    throw new Error(errMessage)
  }

  // Extract filename from Content-Disposition header if available
  let filename = format === 'json' ? `export_${runId}.json` : `report_${runId}.md`
  const disposition = response.headers.get('Content-Disposition')
  if (disposition && disposition.includes('filename=')) {
    const match = disposition.match(/filename="?([^";]+)"?/)
    if (match && match[1]) {
      filename = match[1].trim()
    }
  }

  const blob = await response.blob()
  const blobUrl = window.URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = blobUrl
  link.download = filename
  document.body.appendChild(link)
  link.click()
  document.body.removeChild(link)
  window.URL.revokeObjectURL(blobUrl)
  return { success: true, filename }
}

/**
 * Execute 100% offline deterministic demonstration using synthetic demo data.
 * Zero Gemini LLM calls.
 * @returns {Promise<Object>} complete offline demo results
 */
export async function runOfflineDemo() {
  const response = await fetch(`${BASE_URL}/demo/run-offline`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
  })
  const data = await response.json()
  if (!response.ok) {
    throw new Error(data.error || 'Failed to execute offline demonstration.')
  }
  return data
}

/**
 * Retrieve structured execution event logs for a specific run.
 * @param {string} runId
 * @returns {Promise<{ run_id: string, total_events: number, events: Array }>}
 */
export async function fetchRunLogs(runId) {
  if (!runId) return { run_id: '', total_events: 0, events: [] }
  const response = await fetch(`${BASE_URL}/runs/${encodeURIComponent(runId)}/logs`)
  const data = await response.json()
  if (!response.ok) {
    throw new Error(data.error || `Failed to fetch logs for run '${runId}'.`)
  }
  return data
}

/**
 * Retrieve supervised classification benchmark results and AutoMind Table 3 comparison.
 * Fetches dynamically from backend /api/benchmarks/classification.
 * @returns {Promise<Object>} benchmark data including summary records, automind comparison, config, and citation
 */
export async function fetchClassificationBenchmark() {
  const response = await fetch(`${BASE_URL}/benchmarks/classification`)
  const data = await response.json()
  if (!response.ok) {
    throw new Error(data.error || 'Failed to fetch classification benchmark data.')
  }
  return data
}

/**
 * Retrieve consolidated 5-dataset benchmark results aligning with AutoMind Tables 1-6.
 * Fetches dynamically from backend /api/benchmarks/all-datasets.
 * @returns {Promise<Object>} consolidated benchmark data including summary, automind tables, and per-dataset audit trails
 */
export async function fetchAllDatasetsBenchmark() {
  const response = await fetch(`${BASE_URL}/benchmarks/all-datasets`)
  const data = await response.json()
  if (!response.ok) {
    throw new Error(data.error || 'Failed to fetch consolidated 5-dataset benchmark data.')
  }
  return data
}

/**
 * Get direct URL for benchmark figures served from backend.
 * @param {string} figName
 * @returns {string} figure URL
 */
export function getBenchmarkFigureUrl(figName) {
  return `${BASE_URL}/benchmarks/figure/${figName}`
}

// Re-export client-side research archive & export utilities
export {
  exportRunDirectly,
  exportPairArchive,
  generateRunMarkdown,
  createZipArchive,
  computeSha256,
} from './researchArchive.js'
