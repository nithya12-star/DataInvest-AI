// frontend/tests/research_archive.test.js
// Regression test suite for DataInvest AI Client-Side Export & Research Archive Service.

import { test, describe } from 'node:test'
import assert from 'node:assert/strict'
import {
  sanitizeSlug,
  classifyDatasetCategory,
  computeCrc32,
  computeSha256,
  createZipArchive,
  generateRunMarkdown,
  exportRunDirectly,
  exportPairArchive,
} from '../src/services/researchArchive.js'

describe('Client-Side Research Result Preservation & Export Engine', () => {
  // Mock Run Data fixtures
  const mockBaselineRun = {
    run_id: 'run_20261101_112201_truecsv_baseline_e3a868',
    mode: 'baseline',
    dataset_name: 'True.csv',
    dataset_id: 'True.csv',
    created_at: '2026-11-01T11:22:01.000Z',
    active_phase: 7,
    completed_phases: [1, 2, 3, 7],
    objective: 'Investigate truth consistency in publication metadata.',
    profile: {
      overview: {
        n_rows: 21417,
        n_cols: 4,
        total_missing_cells: 0,
      },
    },
    plan: {
      planner_mode: 'llm',
      planner_model: 'gemini-2.5-flash-lite',
      analysis_tasks: [{ task_id: 'T1' }, { task_id: 'T2' }],
    },
    analysis: {
      results: [
        {
          task_id: 'T1',
          variables: ['subject', 'date'],
          method: 'Chi-Square Contingency Test',
          p_value: 0.0001,
          statistic: 48.21,
        },
      ],
    },
    evaluation: {
      summary: {
        overall_score: 36.4,
        overall_grade: 'D',
      },
      dimensions: [
        { dimension_key: 'statistical_rigor', score: 32.0, weight: 0.35, grade: 'D' },
        { dimension_key: 'truth_grounding', score: 40.0, weight: 0.35, grade: 'C' },
      ],
    },
  }

  const mockProposedRun = {
    run_id: 'run_20261101_112201_truecsv_proposed_e3a868',
    mode: 'proposed',
    dataset_name: 'True.csv',
    dataset_id: 'True.csv',
    created_at: '2026-11-01T11:22:01.000Z',
    active_phase: 7,
    completed_phases: [1, 2, 3, 4, 5, 6, 7],
    objective: 'Investigate truth consistency in publication metadata.',
    profile: {
      overview: {
        n_rows: 21417,
        n_cols: 4,
        total_missing_cells: 0,
      },
    },
    plan: {
      planner_mode: 'llm',
      planner_model: 'gemini-2.5-flash-lite',
      analysis_tasks: [{ task_id: 'T1' }],
    },
    analysis: {
      results: [
        {
          task_id: 'T1',
          variables: ['subject', 'date'],
          method: 'Chi-Square Contingency Test',
          p_value: 0.0001,
          statistic: 48.21,
        },
      ],
    },
    critic: {
      summary: { requires_reinvestigation: 1, accepted: 0 },
      reviews: [
        {
          task_id: 'T1',
          status: 'REQUIRES_REINVESTIGATION',
          confidence: 0.95,
          statistical_validity: 'Sparse contingency cells violate asymptotic assumptions.',
          reason_codes: ['SPARSE_CONTINGENCY_CELLS'],
        },
      ],
    },
    verification: {
      summary: { total_reinvestigated: 1, verified_after_reinvestigation: 1 },
      reinvestigations: [
        {
          task_id: 'T1',
          original_method: 'Chi-Square Test',
          reinvestigated_method: "Fisher's Exact Test / Simulated Permutation",
          final_status: 'VERIFIED',
          methodological_justification: 'Exact permutation test resolved sparse cell assumption violation.',
        },
      ],
    },
    evaluation: {
      summary: {
        overall_score: 84.3,
        overall_grade: 'A',
      },
      dimensions: [
        { dimension_key: 'statistical_rigor', score: 85.0, weight: 0.35, grade: 'A' },
        { dimension_key: 'truth_grounding', score: 83.5, weight: 0.35, grade: 'A' },
      ],
    },
  }

  const mockPairedExperiment = {
    experiment_id: 'exp_20261101_112201_truecsv_e3a868',
    dataset_id: 'True.csv',
    objective: 'Investigate truth consistency in publication metadata.',
    created_at: '2026-11-01T11:22:01.000Z',
    shared_plan: mockProposedRun.plan,
    baseline: {
      run_id: mockBaselineRun.run_id,
      mode: 'baseline',
      overall_score: 36.4,
      overall_grade: 'D',
    },
    proposed: {
      run_id: mockProposedRun.run_id,
      mode: 'proposed',
      overall_score: 84.3,
      overall_grade: 'A',
    },
    comparison: {
      baseline_score: 36.4,
      proposed_score: 84.3,
      overall_score_delta: 47.9,
      assumption_violations_flagged: 1,
      violations_resolved_via_reinvestigation: 1,
      resolution_rate: 1.0,
      scientific_verdict: 'Proposed pipeline improved overall score by +47.9 pts (36.4 -> 84.3).',
    },
    persisted: true,
  }

  test('1. sanitizeSlug produces clean alphanumeric lowercase tokens', () => {
    assert.strictEqual(sanitizeSlug('True.csv'), 'truecsv')
    assert.strictEqual(sanitizeSlug('educational-factors.xlsx'), 'educational_factorsxlsx')
    assert.strictEqual(sanitizeSlug('Run #123 (Baseline/Proposed)!'), 'run_123_baselineproposed')
    assert.strictEqual(sanitizeSlug(''), 'dataset')
  })

  test('2. classifyDatasetCategory accurately maps benchmark datasets to research folders', () => {
    assert.strictEqual(classifyDatasetCategory('Fake.csv'), 'Fake_News')
    assert.strictEqual(classifyDatasetCategory('True.csv'), 'Fake_News')
    assert.strictEqual(classifyDatasetCategory('heart.csv'), 'Heart_Disease')
    assert.strictEqual(classifyDatasetCategory('processed.cleveland.txt'), 'Heart_Disease')
    assert.strictEqual(classifyDatasetCategory('Customers.csv'), 'Customer_Segmentation')
    assert.strictEqual(classifyDatasetCategory('educational-factors.xlsx'), 'Education')
    assert.strictEqual(classifyDatasetCategory('student-mat.csv'), 'Education')
    assert.strictEqual(classifyDatasetCategory('shopmania.csv'), 'ShopMania')
    assert.strictEqual(classifyDatasetCategory('unknown_data.csv'), 'Other_Existing_Datasets')
  })

  test('3. computeSha256 returns valid 64-character lowercase hex digest', async () => {
    const text = 'DataInvest AI Verification Test'
    const hash = await computeSha256(text)
    assert.strictEqual(typeof hash, 'string')
    assert.strictEqual(hash.length, 64)
    assert.match(hash, /^[0-9a-f]{64}$/)

    // Determinism check
    const hash2 = await computeSha256(text)
    assert.strictEqual(hash, hash2)
  })

  test('4. generateRunMarkdown produces structured, valid Markdown without fabrication', () => {
    const md = generateRunMarkdown(mockProposedRun)
    assert.ok(md.includes('# DataInvest AI Scientific Investigation Report — True.csv'))
    assert.ok(md.includes('run_20261101_112201_truecsv_proposed_e3a868'))
    assert.ok(md.includes('84.3 / 100'))
    assert.ok(md.includes('PROPOSED'))
    assert.ok(md.includes('Chi-Square Contingency Test'))
    assert.ok(md.includes('SPARSE_CONTINGENCY_CELLS'))
    assert.ok(md.includes("Fisher's Exact Test / Simulated Permutation"))
    assert.ok(md.includes('Scientific Methodology & Integrity Notice'))
  })

  test('5. exportRunDirectly supports JSON and Markdown with descriptive filenames', async () => {
    const jsonRes = await exportRunDirectly(mockBaselineRun, 'json')
    assert.strictEqual(jsonRes.success, true)
    assert.ok(jsonRes.filename.includes('truecsv_baseline'))
    assert.ok(jsonRes.filename.endsWith('.json'))
    assert.ok(jsonRes.sizeBytes > 500)

    const mdRes = await exportRunDirectly(mockProposedRun, 'markdown')
    assert.strictEqual(mdRes.success, true)
    assert.ok(mdRes.filename.includes('truecsv_proposed'))
    assert.ok(mdRes.filename.endsWith('.md'))
    assert.ok(mdRes.sizeBytes > 500)
  })

  test('6. exportPairArchive creates complete archive with manifest and SHA-256 for valid pair', async () => {
    const archiveRes = await exportPairArchive({
      pairedExperiment: mockPairedExperiment,
      baselineRun: mockBaselineRun,
      proposedRun: mockProposedRun,
    })

    assert.strictEqual(archiveRes.success, true)
    assert.strictEqual(archiveRes.status, 'COMPLETE_PAIR')
    assert.ok(archiveRes.filename.startsWith('DataInvest_Research_Archive_truecsv_'))
    assert.ok(archiveRes.filename.endsWith('.zip'))

    // Verify manifest
    const manifest = archiveRes.manifest
    assert.strictEqual(manifest.pair_status, 'COMPLETE_PAIR')
    assert.strictEqual(manifest.dataset_category, 'Fake_News')
    assert.strictEqual(manifest.comparison_summary.baseline_score, 36.4)
    assert.strictEqual(manifest.comparison_summary.proposed_score, 84.3)
    assert.strictEqual(manifest.runs.length, 2)

    // Check files and checksums
    assert.ok(manifest.archived_files.length >= 6)
    for (const f of manifest.archived_files) {
      assert.ok(f.path.length > 0)
      assert.ok(f.size_bytes > 0)
      assert.strictEqual(f.sha256_checksum.length, 64)
    }
  })

  test('7. exportPairArchive detects incomplete pair and labels status accurately', async () => {
    // Missing baseline
    const incompleteRes = await exportPairArchive({
      pairedExperiment: mockPairedExperiment,
      baselineRun: null,
      proposedRun: mockProposedRun,
    })

    assert.strictEqual(incompleteRes.success, true)
    assert.strictEqual(incompleteRes.status, 'INCOMPLETE_PAIR')
    assert.ok(incompleteRes.manifest.pair_status_details.includes('Baseline run is missing'))
    assert.strictEqual(incompleteRes.manifest.runs.length, 1)
  })

  test('8. createZipArchive produces valid PKZIP format with correct signatures', () => {
    const files = [
      { path: 'test_folder/sample.txt', content: 'Hello DataInvest AI' },
      { path: 'manifest.json', content: '{"status":"ok"}' },
    ]

    const zipBlob = createZipArchive(files)
    assert.ok(zipBlob.size > 100)

    // Convert blob to ArrayBuffer to inspect binary signatures
    return zipBlob.arrayBuffer().then((buf) => {
      const bytes = new Uint8Array(buf)
      const view = new DataView(buf)

      // First 4 bytes must be Local File Header Signature: 0x04034b50 (PK\x03\x04)
      assert.strictEqual(view.getUint32(0, true), 0x04034b50, 'Local Header Signature mismatch')

      // Search for End of Central Directory Signature: 0x06054b50 (PK\x05\x06) in last 64 bytes
      let foundEocd = false
      for (let i = bytes.length - 22; i >= Math.max(0, bytes.length - 100); i--) {
        if (view.getUint32(i, true) === 0x06054b50) {
          foundEocd = true
          break
        }
      }
      assert.ok(foundEocd, 'End of Central Directory signature not found in ZIP buffer')
    })
  })
})
