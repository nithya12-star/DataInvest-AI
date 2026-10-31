// frontend/tests/paired_experiment.test.js
// Offline tests for Paired Experimentation Engine API client & Schema fidelity.
import { test, describe, afterEach } from 'node:test'
import assert from 'node:assert/strict'
import { runPairedExperiment } from '../src/services/api.js'

describe('Paired Experiment API Client & Schema Contract', () => {
  const originalFetch = globalThis.fetch

  afterEach(() => {
    globalThis.fetch = originalFetch
  })

  test('runPairedExperiment sends correct POST request conforming to PairedExperimentRequest', async () => {
    let capturedUrl = null
    let capturedOptions = null

    globalThis.fetch = async (url, options) => {
      capturedUrl = url
      capturedOptions = options
      return {
        ok: true,
        json: async () => ({
          experiment_id: 'exp_test_123',
          dataset_id: 'churn.csv',
          objective: 'Investigate churn factors.',
          created_at: '2026-10-21T17:00:00Z',
          shared_plan: { analysis_tasks: [] },
          baseline: {
            run_id: 'run_baseline_1',
            mode: 'baseline',
            dataset_id: 'churn.csv',
            total_tasks_executed: 4,
            overall_score: 74.0,
            overall_grade: 'B',
            dimension_scores: { statistical_rigor: 70.0 },
            completed_phases: [1, 2, 3, 7],
            has_critic_review: false,
            has_verification: false,
            has_insights: false,
            persisted: true,
            paired_run_id: 'run_proposed_1',
          },
          proposed: {
            run_id: 'run_proposed_1',
            mode: 'proposed',
            dataset_id: 'churn.csv',
            total_tasks_executed: 4,
            overall_score: 86.5,
            overall_grade: 'A',
            dimension_scores: { statistical_rigor: 88.0 },
            completed_phases: [1, 2, 3, 4, 5, 6, 7],
            has_critic_review: true,
            has_verification: true,
            has_insights: true,
            persisted: true,
            paired_run_id: 'run_baseline_1',
          },
          comparison: {
            baseline_score: 74.0,
            proposed_score: 86.5,
            overall_score_delta: 12.5,
            rigor_score_delta: 18.0,
            verification_score_delta: 20.0,
            dimension_deltas: { statistical_rigor: 18.0 },
            assumption_violations_flagged: 2,
            violations_resolved_via_reinvestigation: 2,
            resolution_rate: 1.0,
            total_reinvestigations: 2,
            verified_findings_count: 4,
            questionable_findings_count: 0,
            rejected_findings_count: 0,
            scientific_verdict: 'Proposed pipeline improved score by +12.5 pts.',
          },
          persisted: true,
          partial_persistence: false,
        }),
      }
    }

    const result = await runPairedExperiment({
      datasetId: 'churn.csv',
      objective: 'Investigate churn factors.',
      maxReinvestigations: 1,
      persistRuns: true,
    })

    assert.equal(capturedUrl, '/api/experiments/paired')
    assert.equal(capturedOptions.method, 'POST')
    assert.equal(capturedOptions.headers['Content-Type'], 'application/json')

    const body = JSON.parse(capturedOptions.body)
    assert.equal(body.dataset_id, 'churn.csv')
    assert.equal(body.objective, 'Investigate churn factors.')
    assert.equal(body.max_reinvestigations, 1)
    assert.equal(body.persist_runs, true)

    // Verify response integrity
    assert.equal(result.experiment_id, 'exp_test_123')
    assert.equal(result.baseline.overall_score, 74.0)
    assert.equal(result.proposed.overall_score, 86.5)
    assert.equal(result.comparison.overall_score_delta, 12.5)
    assert.equal(result.comparison.scientific_verdict, 'Proposed pipeline improved score by +12.5 pts.')
  })

  test('runPairedExperiment throws sanitized error on backend failure', async () => {
    globalThis.fetch = async () => ({
      ok: false,
      json: async () => ({ error: 'Dataset not found or unavailable' }),
    })

    await assert.rejects(
      async () => {
        await runPairedExperiment({ datasetId: 'missing.csv' })
      },
      {
        name: 'Error',
        message: 'Dataset not found or unavailable',
      }
    )
  })

  test('runPairedExperiment resolves dataset_id from profile overview if omitted', async () => {
    let capturedBody = null
    globalThis.fetch = async (url, options) => {
      capturedBody = JSON.parse(options.body)
      return {
        ok: true,
        json: async () => ({ experiment_id: 'exp_fallback' }),
      }
    }

    await runPairedExperiment({
      profile: {
        filename: 'dataset_from_profile.csv',
        overview: { n_rows: 100 },
      },
    })

    assert.equal(capturedBody.dataset_id, 'dataset_from_profile.csv')
  })
})
