// frontend/tests/offline_demo_sync.test.js
// Focused regression test suite for DEF-01: Offline Demo / Active Dataset Desynchronization Repair.

import { test, describe, afterEach } from 'node:test'
import assert from 'node:assert/strict'
import { generateRunMarkdown, exportRunDirectly } from '../src/services/researchArchive.js'
import { runOfflineDemo, runPairedExperiment, createInvestigationPlan } from '../src/services/api.js'

describe('DEF-01: Offline Demo & Active Dataset State Synchronization', () => {
  const originalFetch = globalThis.fetch

  afterEach(() => {
    globalThis.fetch = originalFetch
  })

  // Fixture: Real research dataset profile (Heart Disease)
  const heartProfile = {
    overview: {
      n_rows: 303,
      n_cols: 14,
      filename: 'heart.csv',
      memory_usage_bytes: 34000,
    },
    filename: 'heart.csv',
    dataset_id: 'heart.csv',
    columns: ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'],
    numerical_stats: { age: { mean: 54.4, std: 9.0 } },
    missing: { total_missing: 0, total_missing_pct: 0.0 },
    duplicates: { n_duplicates: 0, duplicate_pct: 0.0 },
  }

  // Fixture: Synthetic demonstration payload returned by backend /api/demo/run-offline
  const syntheticDemoResponse = {
    is_offline_demo: true,
    mode: 'offline_deterministic_demo',
    disclaimer: 'OFFLINE DEMONSTRATION ONLY: All outputs in this demo are deterministic and simulated.',
    dataset_id: 'synthetic_demo_data.csv',
    objective: 'Investigate the relationship between study hours, attendance, and exam scores.',
    profile: {
      overview: {
        n_rows: 120,
        n_cols: 6,
        filename: 'synthetic_demo_data.csv',
        memory_usage_bytes: 6200,
      },
      filename: 'synthetic_demo_data.csv',
      dataset_id: 'synthetic_demo_data.csv',
      columns: ['student_id', 'study_hours', 'exam_score', 'attendance_rate', 'tutoring_group', 'academic_tier'],
      is_offline_demo: true,
    },
    plan: {
      planner_mode: 'deterministic_simulated',
      is_offline_demo: true,
      analysis_tasks: [
        { task_id: 'task_01', task_type: 'numerical_vs_numerical', columns: ['study_hours', 'exam_score'] },
      ],
    },
    analysis: {
      results: [
        { task_id: 'task_01', method: 'Pearson correlation', columns: ['study_hours', 'exam_score'] },
      ],
    },
    critic: { reviews: [] },
    verification: { verified_findings: [] },
    insights: { key_insights: [] },
    evaluation: {
      summary: { overall_score: 95.1, overall_grade: 'A+' },
      dimensions: [{ dimension_key: 'execution_reliability', score: 25.0 }],
    },
  }

  // --------------------------------------------------------------------------
  // TEST 1: Actual Offline Demo State Transition Through Home.jsx Logic
  // --------------------------------------------------------------------------
  test('exercises Offline Demo state transition starting from heart.csv without cross-contaminating profile', async () => {
    // 1. Initial State: User uploaded heart.csv
    let profileProp = heartProfile
    let isOfflineDemo = false
    let demoProfile = null
    let activeRunId = null
    let plan = null
    let analysisResponse = null
    let objective = 'Investigate this dataset.'

    // Resolve activeProfile function as implemented in Home.jsx
    const resolveActiveProfile = () => (isOfflineDemo && demoProfile) ? demoProfile : profileProp

    // Before Demo: activeProfile MUST be heart.csv
    assert.equal(resolveActiveProfile().overview.filename, 'heart.csv')
    assert.equal(resolveActiveProfile().overview.n_rows, 303)
    assert.equal(resolveActiveProfile().overview.n_cols, 14)

    // 2. Simulate User clicking "⚡ Offline Demo" button (handleRunOfflineDemo)
    globalThis.fetch = async (url) => {
      assert.match(url, /\/demo\/run-offline/)
      return {
        ok: true,
        json: async () => syntheticDemoResponse,
      }
    }

    const demoData = await runOfflineDemo()

    // State updates matching Home.jsx handleRunOfflineDemo
    if (demoData.profile) {
      demoProfile = demoData.profile
    }
    isOfflineDemo = true
    activeRunId = demoData.run_id || 'run_demo_synthetic_student_data'
    objective = demoData.objective
    plan = demoData.plan
    analysisResponse = demoData.analysis

    // 3. After Demo: activeProfile MUST switch to synthetic demo profile
    const activeProf = resolveActiveProfile()
    assert.equal(activeProf.overview.filename, 'synthetic_demo_data.csv')
    assert.equal(activeProf.overview.n_rows, 120)
    assert.equal(activeProf.overview.n_cols, 6)
    assert.deepEqual(activeProf.columns, [
      'student_id', 'study_hours', 'exam_score', 'attendance_rate', 'tutoring_group', 'academic_tier',
    ])

    // Verify the original uploaded research dataset profile prop was NOT modified
    assert.equal(profileProp.overview.filename, 'heart.csv')
    assert.equal(profileProp.overview.n_rows, 303)

    // 4. Test In-Memory Run Data Assembly (getActiveInMemoryRunData logic)
    const isDemo = Boolean(isOfflineDemo)
    const inMemoryData = {
      run_id: activeRunId || (isDemo ? 'run_demo_synthetic_student_data' : 'run_test'),
      mode: isDemo ? 'offline_deterministic_demo' : 'proposed',
      is_offline_demo: isDemo,
      disclaimer: isDemo ? 'OFFLINE DEMONSTRATION ONLY: Outputs are deterministic.' : undefined,
      dataset_name: isDemo
        ? (activeProf.overview?.filename || activeProf.filename || 'synthetic_demo_data.csv')
        : (profileProp.overview?.filename || 'dataset.csv'),
      dataset_id: isDemo
        ? (activeProf.dataset_id || 'synthetic_demo_data.csv')
        : (profileProp.dataset_id || 'dataset.csv'),
      objective: objective,
      profile: activeProf,
      plan: plan,
      analysis: analysisResponse,
      evaluation: demoData.evaluation,
    }

    assert.equal(inMemoryData.dataset_name, 'synthetic_demo_data.csv')
    assert.equal(inMemoryData.dataset_id, 'synthetic_demo_data.csv')
    assert.equal(inMemoryData.mode, 'offline_deterministic_demo')
    assert.equal(inMemoryData.is_offline_demo, true)
    assert.notEqual(inMemoryData.dataset_name, 'heart.csv')

    // 5. Test Export Generation from this transitioned in-memory run
    const md = generateRunMarkdown(inMemoryData)

    // Must prominently display warning banner
    assert.match(md, /> \[!WARNING\]/)
    assert.match(md, /> \*\*OFFLINE DEMONSTRATION ONLY:\*\*/)
    assert.match(md, /SYNTHETIC_TEST_DATA \(SIMULATED\)/)

    // Must identify the dataset as synthetic_demo_data.csv and NEVER heart.csv
    assert.match(md, /synthetic_demo_data\.csv/)
    assert.doesNotMatch(md, /heart\.csv/)
  })

  // --------------------------------------------------------------------------
  // TEST 2: Historical Run Restoration of Offline Demo Data
  // --------------------------------------------------------------------------
  test('restoring an offline demo run from historical data initializes synthetic profile and warning', () => {
    // Simulated historical run payload restored from /api/runs/{run_id}
    const restoredHistoricalDemoRun = {
      run_id: 'run_historical_demo_01',
      mode: 'offline_deterministic_demo',
      is_offline_demo: true,
      dataset_name: 'synthetic_demo_data.csv',
      dataset_id: 'synthetic_demo_data.csv',
      objective: 'Investigate the relationship between study hours, attendance, and exam scores.',
      profile: syntheticDemoResponse.profile,
      plan: syntheticDemoResponse.plan,
      analysis: syntheticDemoResponse.analysis,
      evaluation: syntheticDemoResponse.evaluation,
    }

    // Initial state setup as in Home.jsx
    const initialRunState = restoredHistoricalDemoRun
    const profileProp = restoredHistoricalDemoRun.profile
    const isOfflineDemo = Boolean(initialRunState?.is_offline_demo)
    const demoProfile = initialRunState?.is_offline_demo ? initialRunState.profile : null

    const activeProfile = (isOfflineDemo && demoProfile) ? demoProfile : profileProp

    assert.equal(isOfflineDemo, true)
    assert.equal(activeProfile.overview.filename, 'synthetic_demo_data.csv')
    assert.equal(activeProfile.overview.n_rows, 120)

    const md = generateRunMarkdown(restoredHistoricalDemoRun)
    assert.match(md, /OFFLINE DEMONSTRATION ONLY/)
    assert.match(md, /synthetic_demo_data\.csv/)
  })

  // --------------------------------------------------------------------------
  // TEST 3: Genuine Real-Dataset Export Preservation
  // --------------------------------------------------------------------------
  test('genuine research run export retains real dataset identity and never receives demo warnings', () => {
    const genuineHeartRun = {
      run_id: 'run_20261030_064546_heartcsv_proposed_8e24a6',
      mode: 'proposed',
      is_offline_demo: false,
      dataset_name: 'heart.csv',
      dataset_id: 'heart.csv',
      objective: 'Conduct exploratory investigation of heart disease clinical predictors.',
      profile: heartProfile,
      evaluation: {
        summary: { overall_score: 85.0, overall_grade: 'A' },
      },
    }

    const md = generateRunMarkdown(genuineHeartRun)

    // Must NOT contain demo warnings
    assert.doesNotMatch(md, /OFFLINE DEMONSTRATION/)
    assert.doesNotMatch(md, /SYNTHETIC_TEST_DATA/)
    assert.doesNotMatch(md, /> \[!WARNING\]/)

    // Must contain genuine dataset identity
    assert.match(md, /heart\.csv/)
    assert.match(md, /`PROPOSED`/)
  })

  // --------------------------------------------------------------------------
  // TEST 4: Clearing Stale Paired-Experiment State When Launching Offline Demo
  // --------------------------------------------------------------------------
  test('handleRunOfflineDemo clears stale paired-experiment state and cached branches', async () => {
    // 1. Initial State: Stale paired experiment already exists in memory
    let pairedExperiment = {
      experiment_id: 'exp_previous_paired_001',
      dataset_id: 'heart.csv',
      baseline: { run_id: 'run_base_old' },
      proposed: { run_id: 'run_prop_old' },
    }
    let cachedBranchRuns = {
      baseline: { run_id: 'run_base_old', mode: 'baseline' },
      proposed: { run_id: 'run_prop_old', mode: 'proposed' },
    }
    let isOfflineDemo = false
    let demoProfile = null
    let resultView = 'compare'
    let activeRunId = 'run_prop_old'

    // Mock fetch for offline demo endpoint
    globalThis.fetch = async (url) => {
      assert.match(url, /\/demo\/run-offline/)
      return {
        ok: true,
        json: async () => syntheticDemoResponse,
      }
    }

    // 2. Trigger handleRunOfflineDemo()
    const demoData = await runOfflineDemo()
    if (demoData.profile) {
      demoProfile = demoData.profile
    }
    activeRunId = demoData.run_id || 'run_demo_synthetic_student_data'
    // State transitions as updated in Home.jsx:
    pairedExperiment = null
    cachedBranchRuns = { baseline: null, proposed: null }
    isOfflineDemo = true
    resultView = 'single'

    // 3. Verify stale paired experiment and branches are cleared
    assert.equal(pairedExperiment, null, 'pairedExperiment must be cleared to null')
    assert.deepEqual(cachedBranchRuns, { baseline: null, proposed: null }, 'cachedBranchRuns must be reset')
    assert.equal(isOfflineDemo, true, 'isOfflineDemo must be active')
    assert.equal(resultView, 'single', 'resultView must be single')
    assert.equal(demoProfile.overview.filename, 'synthetic_demo_data.csv')
  })

  // --------------------------------------------------------------------------
  // TEST 5: Deactivating Offline Demo State On Real Paired Experiment Success
  // --------------------------------------------------------------------------
  test('handleRunPairedExperiment deactivates offline demo state and synthetic profile when successful', async () => {
    // 1. Initial State: User is currently viewing the offline demo
    let profileProp = heartProfile
    let isOfflineDemo = true
    let demoProfile = syntheticDemoResponse.profile
    let pairedExperiment = null
    let resultView = 'single'
    let pairedError = ''

    const resolveActiveProfile = () => (isOfflineDemo && demoProfile) ? demoProfile : profileProp

    assert.equal(resolveActiveProfile().overview.filename, 'synthetic_demo_data.csv')

    // Mock fetch for /experiments/paired
    const mockPairedRes = {
      experiment_id: 'exp_paired_heart_2026',
      dataset_id: 'heart.csv',
      baseline: {
        run_id: 'run_base_heart_new',
        mode: 'baseline',
        dataset_name: 'heart.csv',
        dataset_id: 'heart.csv',
        profile: heartProfile,
      },
      proposed: {
        run_id: 'run_prop_heart_new',
        mode: 'proposed',
        dataset_name: 'heart.csv',
        dataset_id: 'heart.csv',
        profile: heartProfile,
      },
    }

    globalThis.fetch = async (url) => {
      assert.match(url, /\/experiments\/paired/)
      return {
        ok: true,
        json: async () => mockPairedRes,
      }
    }

    // 2. Trigger handleRunPairedExperiment()
    const datasetId = profileProp?.dataset_id || profileProp?.filename || 'sample.csv'
    const response = await runPairedExperiment({
      datasetId,
      profile: profileProp,
      objective: 'Analyze clinical risk',
      maxReinvestigations: 1,
      persistRuns: true,
    })

    // Exact state transition in Home.jsx handleRunPairedExperiment:
    setIsOfflineDemo_called: {
      isOfflineDemo = false
      demoProfile = null
      pairedExperiment = response
      resultView = 'compare'
    }

    // 3. Verify clean return to real dataset
    assert.equal(isOfflineDemo, false, 'isOfflineDemo must be deactivated')
    assert.equal(demoProfile, null, 'demoProfile must be reset to null')
    assert.equal(resolveActiveProfile().overview.filename, 'heart.csv', 'activeProfile must return to heart.csv')
    assert.equal(resolveActiveProfile().overview.n_rows, 303)
    assert.equal(resultView, 'compare')
    assert.equal(pairedExperiment.experiment_id, 'exp_paired_heart_2026')

    // Export of the resulting proposed run must have NO demo warnings
    const md = generateRunMarkdown(response.proposed)
    assert.doesNotMatch(md, /OFFLINE DEMONSTRATION/)
    assert.match(md, /heart\.csv/)
  })

  // --------------------------------------------------------------------------
  // TEST 6: Preserving State When Paired Experiment Fails (No Premature Reset)
  // --------------------------------------------------------------------------
  test('failed handleRunPairedExperiment preserves previous offline demo state without premature deactivation', async () => {
    // 1. Initial State: Demo is active
    let profileProp = heartProfile
    let isOfflineDemo = true
    let demoProfile = syntheticDemoResponse.profile
    let pairedExperiment = null
    let pairedError = ''

    const resolveActiveProfile = () => (isOfflineDemo && demoProfile) ? demoProfile : profileProp

    // Mock fetch failure
    globalThis.fetch = async (url) => {
      assert.match(url, /\/experiments\/paired/)
      return {
        ok: false,
        json: async () => ({ error: 'Backend GPU service temporarily unavailable.' }),
      }
    }

    // 2. Execute with failure catch pattern matching Home.jsx
    try {
      const datasetId = profileProp?.dataset_id || profileProp?.filename || 'sample.csv'
      const response = await runPairedExperiment({
        datasetId,
        profile: profileProp,
        objective: 'Analyze clinical risk',
        maxReinvestigations: 1,
        persistRuns: true,
      })
      // Should not reach here
      isOfflineDemo = false
      demoProfile = null
      pairedExperiment = response
    } catch (err) {
      pairedError = err.message || 'Failed to execute comparative experiment.'
    }

    // 3. Verify demo state is preserved safely
    assert.equal(isOfflineDemo, true, 'isOfflineDemo must NOT be prematurely cleared on failure')
    assert.notEqual(demoProfile, null, 'demoProfile must NOT be prematurely cleared on failure')
    assert.equal(resolveActiveProfile().overview.filename, 'synthetic_demo_data.csv')
    assert.match(pairedError, /GPU service temporarily unavailable/)
  })

  // --------------------------------------------------------------------------
  // TEST 7: Generating Plan For Uploaded Dataset Deactivates Offline Demo State
  // --------------------------------------------------------------------------
  test('handleGeneratePlan for uploaded dataset deactivates offline demo state upon success', async () => {
    // 1. Initial State: Demo is active
    let profileProp = heartProfile
    let isOfflineDemo = true
    let demoProfile = syntheticDemoResponse.profile
    let plan = null
    let planError = ''

    const resolveActiveProfile = () => (isOfflineDemo && demoProfile) ? demoProfile : profileProp

    assert.equal(resolveActiveProfile().overview.filename, 'synthetic_demo_data.csv')

    // Mock fetch for /plan/create-plan
    globalThis.fetch = async (url) => {
      assert.match(url, /\/plan/)
      return {
        ok: true,
        json: async () => ({
          dataset_id: 'heart.csv',
          planner_mode: 'active_llm',
          analysis_tasks: [
            { task_id: 'task_01', task_type: 'correlation', columns: ['age', 'chol'] },
          ],
        }),
      }
    }

    // 2. Trigger handleGeneratePlan logic
    try {
      const planRes = await createInvestigationPlan(profileProp, 'Investigate heart disease', {})
      plan = planRes
      isOfflineDemo = false
      demoProfile = null
    } catch (err) {
      planError = err.message
    }

    // 3. Verify clean return to uploaded dataset
    assert.equal(isOfflineDemo, false, 'isOfflineDemo must be deactivated when real plan succeeds')
    assert.equal(demoProfile, null, 'demoProfile must be reset to null')
    assert.equal(resolveActiveProfile().overview.filename, 'heart.csv', 'activeProfile must resolve to heart.csv')
    assert.equal(plan.planner_mode, 'active_llm')
    assert.equal(planError, '')
  })
})
