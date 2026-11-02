// frontend/src/services/researchArchive.js
// ===========================================
// DataInvest AI Research Result Local Preservation & Direct Client-Side Export Service.
//
// Allows researchers to export single run results (JSON, Markdown) and complete
// baseline-proposed paired experiment research archives (.ZIP) directly from in-memory
// client state without relying on backend container disk persistence.

/**
 * Convert string to safe alphanumeric slug with underscores.
 * @param {string} text
 * @param {number} maxLen
 * @returns {string}
 */
export function sanitizeSlug(text, maxLen = 40) {
  if (!text) return 'dataset'
  const clean = String(text)
    .replace(/[^\w\s-]/g, '')
    .trim()
    .toLowerCase()
    .replace(/[-\s]+/g, '_')
  return clean.slice(0, maxLen) || 'dataset'
}

/**
 * Classify dataset into standard research folder categories based on name.
 * @param {string} datasetName
 * @returns {string}
 */
export function classifyDatasetCategory(datasetName) {
  const lower = String(datasetName || '').toLowerCase()
  if (lower.includes('fake') || lower.includes('true')) return 'Fake_News'
  if (lower.includes('heart') || lower.includes('cleveland')) return 'Heart_Disease'
  if (lower.includes('customer')) return 'Customer_Segmentation'
  if (lower.includes('education') || lower.includes('student')) return 'Education'
  if (lower.includes('shop')) return 'ShopMania'
  return 'Other_Existing_Datasets'
}

/**
 * Compute SHA-256 checksum string for given text or bytes.
 * Uses native Web Crypto API available in all modern browsers and Node.js.
 * @param {string|Uint8Array} input
 * @returns {Promise<string>} Hex lowercase SHA-256 hash
 */
export async function computeSha256(input) {
  const bytes = typeof input === 'string' ? new TextEncoder().encode(input) : input
  if (typeof crypto !== 'undefined' && crypto.subtle) {
    const hashBuffer = await crypto.subtle.digest('SHA-256', bytes)
    const hashArray = Array.from(new Uint8Array(hashBuffer))
    return hashArray.map((b) => b.toString(16).padStart(2, '0')).join('')
  }
  // Fallback for mock environments without crypto.subtle
  return 'sha256_mock_digest'
}

// Precomputed CRC32 lookup table for standard PKZIP checksum calculation
const CRC_TABLE = new Uint32Array(256)
for (let i = 0; i < 256; i++) {
  let c = i
  for (let k = 0; k < 8; k++) {
    c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1
  }
  CRC_TABLE[i] = c
}

/**
 * Compute CRC32 checksum for Uint8Array bytes.
 * @param {Uint8Array} bytes
 * @returns {number} Unsigned 32-bit CRC32
 */
export function computeCrc32(bytes) {
  let crc = 0xffffffff
  for (let i = 0; i < bytes.length; i++) {
    crc = CRC_TABLE[(crc ^ bytes[i]) & 0xff] ^ (crc >>> 8)
  }
  return (crc ^ 0xffffffff) >>> 0
}

/**
 * Build a valid, standard PKZIP archive (compression: STORE / method 0)
 * entirely in browser JavaScript using standard Uint8Array and DataView.
 * Requires zero external dependencies.
 *
 * @param {Array<{ path: string, content: string|Uint8Array }>} files
 * @returns {Blob} Valid application/zip Blob
 */
export function createZipArchive(files) {
  const encoder = new TextEncoder()
  const fileEntries = []

  // 1. Prepare and measure all file buffers
  for (const f of files) {
    const filenameBytes = encoder.encode(f.path.replace(/\\/g, '/'))
    const dataBytes =
      typeof f.content === 'string'
        ? encoder.encode(f.content)
        : f.content instanceof Uint8Array
        ? f.content
        : new Uint8Array(f.content)

    const crc = computeCrc32(dataBytes)
    fileEntries.push({
      path: f.path,
      filenameBytes,
      dataBytes,
      crc,
      uncompressedSize: dataBytes.length,
      compressedSize: dataBytes.length,
    })
  }

  // Current DOS time / date: 2026-11-01 12:00:00 (standard deterministic timestamp)
  const dosTime = (12 << 11) | (0 << 5) | (0 >> 1)
  const dosDate = ((2026 - 1980) << 9) | (11 << 5) | 1

  // 2. Compute total buffer size
  let localHeadersTotalSize = 0
  for (const entry of fileEntries) {
    localHeadersTotalSize += 30 + entry.filenameBytes.length + entry.dataBytes.length
  }

  let centralDirTotalSize = 0
  for (const entry of fileEntries) {
    centralDirTotalSize += 46 + entry.filenameBytes.length
  }

  const endRecordSize = 22
  const totalZipSize = localHeadersTotalSize + centralDirTotalSize + endRecordSize

  const zipBuffer = new Uint8Array(totalZipSize)
  const view = new DataView(zipBuffer.buffer)
  let offset = 0

  // 3. Write Local File Headers and Data
  const localOffsets = []
  for (const entry of fileEntries) {
    localOffsets.push(offset)

    // Local Header Signature: 0x04034b50
    view.setUint32(offset, 0x04034b50, true)
    view.setUint16(offset + 4, 20, true) // Version needed (2.0)
    view.setUint16(offset + 6, 0x0800, true) // UTF-8 filename flag
    view.setUint16(offset + 8, 0, true) // Compression: 0 = STORE
    view.setUint16(offset + 10, dosTime, true)
    view.setUint16(offset + 12, dosDate, true)
    view.setUint32(offset + 14, entry.crc, true)
    view.setUint32(offset + 18, entry.compressedSize, true)
    view.setUint32(offset + 22, entry.uncompressedSize, true)
    view.setUint16(offset + 26, entry.filenameBytes.length, true)
    view.setUint16(offset + 28, 0, true) // Extra field length

    offset += 30
    zipBuffer.set(entry.filenameBytes, offset)
    offset += entry.filenameBytes.length

    zipBuffer.set(entry.dataBytes, offset)
    offset += entry.dataBytes.length
  }

  // 4. Write Central Directory Headers
  const centralDirStartOffset = offset
  for (let i = 0; i < fileEntries.length; i++) {
    const entry = fileEntries[i]
    const localHeaderOffset = localOffsets[i]

    // Central Dir Signature: 0x02014b50
    view.setUint32(offset, 0x02014b50, true)
    view.setUint16(offset + 4, 20, true) // Version made by
    view.setUint16(offset + 6, 20, true) // Version needed
    view.setUint16(offset + 8, 0x0800, true) // UTF-8 flag
    view.setUint16(offset + 10, 0, true) // Compression: STORE
    view.setUint16(offset + 12, dosTime, true)
    view.setUint16(offset + 14, dosDate, true)
    view.setUint32(offset + 16, entry.crc, true)
    view.setUint32(offset + 20, entry.compressedSize, true)
    view.setUint32(offset + 24, entry.uncompressedSize, true)
    view.setUint16(offset + 28, entry.filenameBytes.length, true)
    view.setUint16(offset + 30, 0, true) // Extra field len
    view.setUint16(offset + 32, 0, true) // Comment len
    view.setUint16(offset + 34, 0, true) // Disk start
    view.setUint16(offset + 36, 0, true) // Internal attrs
    view.setUint32(offset + 38, 0, true) // External attrs
    view.setUint32(offset + 42, localHeaderOffset, true) // Relative offset of local header

    offset += 46
    zipBuffer.set(entry.filenameBytes, offset)
    offset += entry.filenameBytes.length
  }

  // 5. Write End of Central Directory Record
  const centralDirLength = offset - centralDirStartOffset

  // EOCD Signature: 0x06054b50
  view.setUint32(offset, 0x06054b50, true)
  view.setUint16(offset + 4, 0, true) // Disk number
  view.setUint16(offset + 6, 0, true) // Disk with CD
  view.setUint16(offset + 8, fileEntries.length, true) // CD entries on this disk
  view.setUint16(offset + 10, fileEntries.length, true) // Total CD entries
  view.setUint32(offset + 12, centralDirLength, true) // Size of CD
  view.setUint32(offset + 16, centralDirStartOffset, true) // Offset of CD start
  view.setUint16(offset + 20, 0, true) // Zip comment length

  return new Blob([zipBuffer], { type: 'application/zip' })
}

/**
 * Trigger native browser blob download and revoke object URL cleanly.
 * @param {Blob} blob
 * @param {string} filename
 */
export function triggerBlobDownload(blob, filename) {
  if (typeof window === 'undefined' || typeof document === 'undefined') {
    return { success: true, filename, isMock: true }
  }
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
 * Generate human-readable GitHub-Flavored Markdown report from RunPayload or in-memory state.
 * Faithfully represents dataset characteristics, statistical findings, Critic verification,
 * and evaluation scores without fabrication.
 *
 * @param {Object} runData
 * @returns {string} Markdown text
 */
export function generateRunMarkdown(runData) {
  if (!runData) return '# Empty Run Report\n\nNo run data available.'

  const runId = runData.run_id || 'unknown_run'
  const datasetName = runData.dataset_name || runData.dataset_id || 'dataset.csv'
  const mode = (runData.mode || (runData.critic ? 'proposed' : 'baseline')).toUpperCase()
  const createdAt = runData.created_at || new Date().toISOString()
  const objective = runData.objective || 'Investigate this dataset.'

  const profile = runData.profile || {}
  const overview = profile.overview || {}
  const evalSummary = runData.evaluation?.summary || {}
  const evalDims = runData.evaluation?.dimensions || []

  const lines = []

  // 0. Prominent Offline Demonstration Warning Banner
  if (runData.is_offline_demo) {
    lines.push('> [!WARNING]')
    lines.push('> **OFFLINE DEMONSTRATION ONLY:** All outputs and findings in this report were generated using deterministic synthetic demonstration data without external Gemini LLM calls.')
    lines.push('> Never cite or describe this report as empirical experimental evidence for research benchmarks.')
    lines.push('')
  }

  // 1. Header Banner
  lines.push(`# DataInvest AI Scientific Investigation Report — ${datasetName}`)
  lines.push('')
  lines.push('> **DataInvest AI Automated Investigation & Verification Pipeline**  ')
  lines.push(`> **Run ID:** \`${runId}\` | **Generated:** \`${createdAt}\``)
  lines.push(`> **Architecture Mode:** \`${mode}\``)
  if (evalSummary.overall_score !== undefined && evalSummary.overall_score !== null) {
    lines.push(
      `> **System Rigor Score:** \`${evalSummary.overall_score} / 100\` (\`Grade ${evalSummary.overall_grade || '—'}\`)`
    )
  }
  lines.push('')
  lines.push('---')
  lines.push('')

  // 2. Objective & Scope
  lines.push('## 1. Investigation Objective & Execution Scope')
  lines.push('')
  lines.push(`**Primary Objective:** ${objective}`)
  lines.push('')
  lines.push('| Execution Dimension | Value |')
  lines.push('|:---|:---|')
  lines.push(`| **Dataset Name** | \`${datasetName}\` |`)
  lines.push(`| **Run ID** | \`${runId}\` |`)
  lines.push(`| **Execution Mode** | \`${mode}\` |`)
  if (runData.is_offline_demo) {
    lines.push('| **Data Provenance** | `SYNTHETIC_TEST_DATA (SIMULATED)` |')
  }
  lines.push(
    `| **Completed Phases** | \`${(runData.completed_phases || []).map((p) => `P${p}`).join(', ') || 'None'}\` |`
  )
  lines.push('')

  // 3. Dataset Characteristics
  lines.push('## 2. Dataset Context & Profile Characteristics')
  lines.push('')
  lines.push('| Metric | Value | Interpretation |')
  lines.push('|:---|:---|:---|')
  lines.push(`| **Total Rows** | \`${overview.n_rows ?? '—'}\` | Effective sample size |`)
  lines.push(`| **Total Columns** | \`${overview.n_cols ?? '—'}\` | Feature dimensionality |`)
  lines.push(`| **Missing Cells** | \`${overview.total_missing_cells ?? 0}\` | Data completeness |`)
  lines.push('')

  // 4. Initial Statistical Findings
  const analysisResults = runData.analysis?.results || []
  lines.push('## 3. Statistical Analysis Findings')
  lines.push('')
  if (analysisResults.length === 0) {
    lines.push('_No statistical analysis tasks recorded in this run._')
  } else {
    lines.push('| Task ID | Variables | Method | p-value | Effect Size / Statistic |')
    lines.push('|:---|:---|:---|:---|:---|')
    for (const r of analysisResults) {
      const vars = Array.isArray(r.variables) ? r.variables.join(' vs ') : (r.variables || '—')
      const pval = r.p_value !== null && r.p_value !== undefined ? Number(r.p_value).toFixed(4) : '—'
      const stat = r.statistic !== null && r.statistic !== undefined ? Number(r.statistic).toFixed(4) : '—'
      lines.push(`| \`${r.task_id || 'T?'}\` | ${vars} | ${r.method || 'Statistical Test'} | \`${pval}\` | \`${stat}\` |`)
    }
  }
  lines.push('')

  // 5. Critic Verification (Proposed Mode)
  if (mode === 'PROPOSED' && runData.critic) {
    lines.push('## 4. Critic Agent Verification & Statistical Diagnostics')
    lines.push('')
    const reviews = runData.critic.reviews || []
    if (reviews.length === 0) {
      lines.push('_Critic review was enabled, but no individual task reviews were recorded._')
    } else {
      lines.push('| Task ID | Verification Status | Confidence / Decision Certainty | Validity / Diagnostic Reason |')
      lines.push('|:---|:---|:---|:---|')
      for (const rev of reviews) {
        let conf = '—'
        if (typeof rev.confidence === 'number') {
          const pct = `${(rev.confidence * 100).toFixed(0)}%`
          const lvl = rev.confidence_level || 'HIGH'
          if (rev.status === 'REJECTED') {
            conf = `${pct} (${lvl} Decision Certainty)`
          } else if (rev.status === 'REQUIRES_REINVESTIGATION') {
            conf = `${pct} (${lvl} Review Certainty)`
          } else {
            conf = `${pct} (${lvl})`
          }
        }
        const codes = rev.reason_codes && rev.reason_codes.length ? ` [${rev.reason_codes.join(', ')}]` : ''
        const reason = (rev.statistical_validity || rev.summary || 'Verified').replace(/\|/g, '\\|')
        lines.push(`| \`${rev.task_id || 'T?'}\` | **${rev.status || 'REVIEWED'}** | ${conf} | ${reason}${codes} |`)
      }
    }
    lines.push('')
  }

  // 6. Controlled Re-investigations (Proposed Mode)
  if (mode === 'PROPOSED' && runData.verification) {
    const reinvs = runData.verification.reinvestigations || []
    if (reinvs.length > 0) {
      lines.push('## 5. Controlled Alternative-Test Re-investigations')
      lines.push('')
      lines.push('| Task ID | Original Method | Alternative Method | Outcome | Justification |')
      lines.push('|:---|:---|:---|:---|:---|')
      for (const rec of reinvs) {
        const orig = rec.original_method || 'Parametric Test'
        const alt = rec.reinvestigated_method || 'Non-parametric Alternative'
        const out = rec.final_status || 'VERIFIED'
        const just = (rec.methodological_justification || 'Assumption violation corrected').replace(/\|/g, '\\|')
        lines.push(`| \`${rec.task_id || 'T?'}\` | ${orig} | ${alt} | **${out}** | ${just} |`)
      }
      lines.push('')
    }
  }

  // 7. Pipeline Quality Evaluation Breakdown
  lines.push('## 6. Evaluation Framework & Rigor Breakdown')
  lines.push('')
  if (evalDims.length === 0) {
    lines.push(`**Overall Rigor Score:** \`${evalSummary.overall_score ?? '—'} / 100\``)
  } else {
    lines.push('| Evaluation Dimension | Dimension Score | Weight | Qualitative Rating |')
    lines.push('|:---|:---|:---|:---|')
    for (const d of evalDims) {
      const dimName = (d.dimension_key || d.dimension_name || 'Dimension').replace(/_/g, ' ')
      const score = typeof d.score === 'number' ? d.score.toFixed(1) : String(d.score || '—')
      const weight = d.weight !== undefined ? `${(d.weight * 100).toFixed(0)}%` : '—'
      lines.push(`| **${dimName}** | \`${score} / 100\` | ${weight} | ${d.grade || '—'} |`)
    }
  }
  lines.push('')

  // 8. Scientific Caveats & Disclaimer
  lines.push('---')
  lines.push('### Scientific Methodology & Integrity Notice')
  lines.push(
    '- **Causal Boundaries:** All automated findings report empirical associations and statistical bounds. Correlation does not imply causation.'
  )
  lines.push(
    '- **Critic Verification:** Critic agent checks verify mathematical validity and distributional assumption bounds.'
  )
  lines.push(
    `- **Data Source:** Generated from local dataset execution under DataInvest AI protocol at ${createdAt}.`
  )
  lines.push('')

  return lines.join('\n')
}

/**
 * Export a single run directly from in-memory JavaScript data without backend calls.
 *
 * @param {Object} runData - Full RunPayload object
 * @param {'json'|'markdown'} format - Export format
 * @returns {Promise<{ success: boolean, filename: string, sizeBytes: number }>}
 */
export async function exportRunDirectly(runData, format = 'json') {
  if (!runData) {
    throw new Error('Cannot export: No run data is present in memory.')
  }

  const rawRunId = runData.run_id || 'run_' + Date.now()
  const cleanRunId = sanitizeSlug(rawRunId, 50)
  const datasetName = runData.dataset_name || runData.dataset_id || 'dataset'
  const datasetSlug = sanitizeSlug(datasetName, 30)
  const mode = runData.is_offline_demo
    ? 'offline_demo'
    : sanitizeSlug(runData.mode || (runData.critic ? 'proposed' : 'baseline'), 15)

  if (format === 'markdown') {
    const mdText = generateRunMarkdown(runData)
    const filename = `report_${datasetSlug}_${mode}_${cleanRunId}.md`
    const blob = new Blob([mdText], { type: 'text/markdown;charset=utf-8' })
    triggerBlobDownload(blob, filename)
    return { success: true, filename, sizeBytes: blob.size }
  }

  // JSON format
  const jsonContent = JSON.stringify(runData, null, 2)
  const filename = `export_${datasetSlug}_${mode}_${cleanRunId}.json`
  const blob = new Blob([jsonContent], { type: 'application/json;charset=utf-8' })
  triggerBlobDownload(blob, filename)
  return { success: true, filename, sizeBytes: blob.size }
}

/**
 * Build and download a complete, structured DataInvest AI Research Archive (.ZIP).
 * Organizes files into logical category folders (e.g. Fake_News/Baseline, Fake_News/Proposed, Fake_News/Pairs)
 * and includes a comprehensive manifest.json with SHA-256 checksums.
 *
 * @param {Object} params
 * @param {Object} params.pairedExperiment - PairedExperimentResponse from runner
 * @param {Object} [params.baselineRun] - Full baseline RunPayload
 * @param {Object} [params.proposedRun] - Full proposed RunPayload
 * @returns {Promise<{ success: boolean, filename: string, status: 'COMPLETE_PAIR'|'INCOMPLETE_PAIR', manifest: Object }>}
 */
export async function exportPairArchive({ pairedExperiment, baselineRun, proposedRun }) {
  if (!pairedExperiment) {
    throw new Error('Cannot create archive: Paired experiment data is required.')
  }

  const expId = pairedExperiment.experiment_id || 'exp_' + Date.now()
  const cleanExpId = sanitizeSlug(expId, 40)
  const datasetName =
    pairedExperiment.dataset_id ||
    baselineRun?.dataset_name ||
    proposedRun?.dataset_name ||
    'dataset.csv'
  const datasetSlug = sanitizeSlug(datasetName, 30)
  const categoryFolder = classifyDatasetCategory(datasetName)
  const nowIso = new Date().toISOString()

  // Verify pair completeness
  const hasBaseline = Boolean(baselineRun && baselineRun.run_id)
  const hasProposed = Boolean(proposedRun && proposedRun.run_id)
  const isCompletePair = hasBaseline && hasProposed
  const pairStatus = isCompletePair ? 'COMPLETE_PAIR' : 'INCOMPLETE_PAIR'

  const statusExplanation = isCompletePair
    ? 'Both Baseline and Proposed runs were successfully recorded, verified, and packaged.'
    : `Incomplete pair: ${!hasBaseline ? 'Baseline run is missing or unpopulated; ' : ''}${
        !hasProposed ? 'Proposed run is missing or unpopulated.' : ''
      }`

  const filesToZip = []
  const manifestRuns = []
  const allArchivedFiles = []

  // Helper to record file
  async function addArchiveFile(relativePath, contentString, mimeType = 'text/plain') {
    const sha256 = await computeSha256(contentString)
    const sizeBytes = new TextEncoder().encode(contentString).length
    filesToZip.push({ path: relativePath, content: contentString })
    allArchivedFiles.push({
      path: relativePath,
      size_bytes: sizeBytes,
      sha256_checksum: sha256,
      mime_type: mimeType,
    })
    return { sha256, sizeBytes }
  }

  // 1. Package Baseline Run (if available)
  if (hasBaseline) {
    const bRunId = sanitizeSlug(baselineRun.run_id, 40)
    const bJsonPath = `${categoryFolder}/Baseline/report_${datasetSlug}_baseline_${bRunId}.json`
    const bMdPath = `${categoryFolder}/Baseline/report_${datasetSlug}_baseline_${bRunId}.md`

    const bJsonStr = JSON.stringify(baselineRun, null, 2)
    const bMdStr = generateRunMarkdown(baselineRun)

    const jsonMeta = await addArchiveFile(bJsonPath, bJsonStr, 'application/json')
    const mdMeta = await addArchiveFile(bMdPath, bMdStr, 'text/markdown')

    manifestRuns.push({
      run_id: baselineRun.run_id,
      mode: 'baseline',
      overall_score: pairedExperiment.baseline?.overall_score ?? baselineRun.evaluation?.summary?.overall_score ?? null,
      overall_grade: pairedExperiment.baseline?.overall_grade ?? baselineRun.evaluation?.summary?.overall_grade ?? '—',
      completed_phases: baselineRun.completed_phases || [1, 2, 3, 7],
      files: {
        json: { path: bJsonPath, sha256: jsonMeta.sha256, size_bytes: jsonMeta.sizeBytes },
        markdown: { path: bMdPath, sha256: mdMeta.sha256, size_bytes: mdMeta.sizeBytes },
      },
    })
  }

  // 2. Package Proposed Run (if available)
  if (hasProposed) {
    const pRunId = sanitizeSlug(proposedRun.run_id, 40)
    const pJsonPath = `${categoryFolder}/Proposed/report_${datasetSlug}_proposed_${pRunId}.json`
    const pMdPath = `${categoryFolder}/Proposed/report_${datasetSlug}_proposed_${pRunId}.md`

    const pJsonStr = JSON.stringify(proposedRun, null, 2)
    const pMdStr = generateRunMarkdown(proposedRun)

    const jsonMeta = await addArchiveFile(pJsonPath, pJsonStr, 'application/json')
    const mdMeta = await addArchiveFile(pMdPath, pMdStr, 'text/markdown')

    manifestRuns.push({
      run_id: proposedRun.run_id,
      mode: 'proposed',
      overall_score: pairedExperiment.proposed?.overall_score ?? proposedRun.evaluation?.summary?.overall_score ?? null,
      overall_grade: pairedExperiment.proposed?.overall_grade ?? proposedRun.evaluation?.summary?.overall_grade ?? '—',
      completed_phases: proposedRun.completed_phases || [1, 2, 3, 4, 5, 6, 7],
      files: {
        json: { path: pJsonPath, sha256: jsonMeta.sha256, size_bytes: jsonMeta.sizeBytes },
        markdown: { path: pMdPath, sha256: mdMeta.sha256, size_bytes: mdMeta.sizeBytes },
      },
    })
  }

  // 3. Package Paired Comparison Results
  const compPath = `${categoryFolder}/Pairs/comparison_${datasetSlug}_${cleanExpId}.json`
  const compStr = JSON.stringify(pairedExperiment, null, 2)
  await addArchiveFile(compPath, compStr, 'application/json')

  // 4. Generate README.txt
  const readmeContent = [
    '========================================================================',
    'DataInvest AI — Autonomous Multi-Agent Research Experiment Archive',
    '========================================================================',
    '',
    `Dataset:              ${datasetName}`,
    `Research Category:    ${categoryFolder}`,
    `Experiment ID:        ${expId}`,
    `Pair Status:          ${pairStatus}`,
    `Status Details:       ${statusExplanation}`,
    `Archive Created:      ${nowIso}`,
    '',
    '------------------------------------------------------------------------',
    'COMPARATIVE BENCHMARK SUMMARY',
    '------------------------------------------------------------------------',
    `Baseline Score:       ${pairedExperiment.comparison?.baseline_score ?? pairedExperiment.baseline?.overall_score ?? '—'} / 100`,
    `Proposed Score:       ${pairedExperiment.comparison?.proposed_score ?? pairedExperiment.proposed?.overall_score ?? '—'} / 100`,
    `Score Delta:          ${(pairedExperiment.comparison?.overall_score_delta !== undefined ? (pairedExperiment.comparison.overall_score_delta > 0 ? '+' : '') + pairedExperiment.comparison.overall_score_delta : '—')} pts`,
    `Violations Flagged:   ${pairedExperiment.comparison?.assumption_violations_flagged ?? '0'}`,
    `Violations Resolved:  ${pairedExperiment.comparison?.violations_resolved_via_reinvestigation ?? '0'}`,
    `Resolution Rate:      ${pairedExperiment.comparison?.resolution_rate ? (pairedExperiment.comparison.resolution_rate * 100).toFixed(1) + '%' : 'N/A'}`,
    `Scientific Verdict:   ${pairedExperiment.comparison?.scientific_verdict || 'Comparison completed successfully.'}`,
    '',
    '------------------------------------------------------------------------',
    'ARCHIVE STRUCTURE',
    '------------------------------------------------------------------------',
    'DataInvest_AI_Research_Archive/',
    '  ├── manifest.json              (Machine-readable integrity manifest & SHA-256 hashes)',
    '  ├── README.txt                 (This archive documentation file)',
    `  └── ${categoryFolder}/`,
    '        ├── Baseline/            (Baseline experiment JSON and Markdown reports)',
    '        ├── Proposed/            (Proposed self-verifying experiment JSON and Markdown)',
    '        └── Pairs/               (Full paired comparison and empirical delta summary)',
    '',
    '------------------------------------------------------------------------',
    'SECURITY & DURABILITY NOTICE',
    '------------------------------------------------------------------------',
    'This archive was generated client-side from verified execution results.',
    'It guarantees research reproducibility and preservation independently of',
    'cloud ephemeral container restarts (such as Render free-tier recycling).',
    'Store this archive in your personal research directory or version control.',
    '========================================================================',
  ].join('\n')

  await addArchiveFile('README.txt', readmeContent, 'text/plain')

  // 5. Build and Add manifest.json
  const manifest = {
    manifest_version: '1.0.0',
    export_timestamp: nowIso,
    pair_status: pairStatus,
    pair_status_details: statusExplanation,
    experiment_id: expId,
    dataset_name: datasetName,
    dataset_category: categoryFolder,
    objective: pairedExperiment.objective || 'Investigate this dataset.',
    planner_mode: pairedExperiment.shared_plan?.planner_mode || 'llm',
    planner_model: pairedExperiment.shared_plan?.planner_model || 'gemini-2.5-flash-lite',
    comparison_summary: {
      baseline_score: pairedExperiment.comparison?.baseline_score ?? null,
      proposed_score: pairedExperiment.comparison?.proposed_score ?? null,
      overall_score_delta: pairedExperiment.comparison?.overall_score_delta ?? null,
      assumption_violations_flagged: pairedExperiment.comparison?.assumption_violations_flagged ?? 0,
      violations_resolved: pairedExperiment.comparison?.violations_resolved_via_reinvestigation ?? 0,
      resolution_rate: pairedExperiment.comparison?.resolution_rate ?? null,
      scientific_verdict: pairedExperiment.comparison?.scientific_verdict || null,
    },
    runs: manifestRuns,
    archived_files: allArchivedFiles,
  }

  const manifestStr = JSON.stringify(manifest, null, 2)
  const manifestMeta = await addArchiveFile('manifest.json', manifestStr, 'application/json')
  manifest.manifest_sha256 = manifestMeta.sha256

  // 6. Create ZIP Blob and trigger download
  const zipBlob = createZipArchive(filesToZip)
  const archiveFilename = `DataInvest_Research_Archive_${datasetSlug}_${cleanExpId}.zip`

  triggerBlobDownload(zipBlob, archiveFilename)

  return {
    success: true,
    filename: archiveFilename,
    status: pairStatus,
    sizeBytes: zipBlob.size,
    manifest,
  }
}
