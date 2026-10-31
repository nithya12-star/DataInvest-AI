// frontend/tests/stage_c.test.js
// Stage C offline test suite for comparison metrics, offline demo API, and base-paper citations.
import { test, describe, afterEach } from 'node:test'
import assert from 'node:assert/strict'
import { runOfflineDemo, fetchRunLogs, fetchClassificationBenchmark } from '../src/services/api.js'

describe('Stage C: Offline Demo & Logging API Client', () => {
  const originalFetch = globalThis.fetch

  afterEach(() => {
    globalThis.fetch = originalFetch
  })

  test('runOfflineDemo sends POST request to /api/demo/run-offline without LLM calls', async () => {
    let capturedUrl = null
    let capturedOptions = null

    globalThis.fetch = async (url, options) => {
      capturedUrl = url
      capturedOptions = options
      return {
        ok: true,
        json: async () => ({
          is_offline_demo: true,
          mode: 'offline_deterministic_demo',
          disclaimer: 'OFFLINE DEMONSTRATION ONLY: Outputs are deterministic and simulated without Gemini API calls.',
          dataset_id: 'synthetic_demo_data.csv',
          plan: { planner_mode: 'deterministic_simulated' },
        }),
      }
    }

    const result = await runOfflineDemo()

    assert.equal(capturedUrl, '/api/demo/run-offline')
    assert.equal(capturedOptions.method, 'POST')
    assert.equal(result.is_offline_demo, true)
    assert.equal(result.mode, 'offline_deterministic_demo')
    assert.match(result.disclaimer, /OFFLINE DEMONSTRATION/)
  })

  test('fetchRunLogs fetches structured event logs for a given runId', async () => {
    let capturedUrl = null

    globalThis.fetch = async (url) => {
      capturedUrl = url
      return {
        ok: true,
        json: async () => ({
          run_id: 'run_20261023_test_001',
          total_events: 2,
          events: [
            { event_type: 'PLANNER_COMPLETED', outcome: 'SUCCESS' },
            { event_type: 'CRITIC_COMPLETED', outcome: 'PASSED' },
          ],
        }),
      }
    }

    const result = await fetchRunLogs('run_20261023_test_001')

    assert.equal(capturedUrl, '/api/runs/run_20261023_test_001/logs')
    assert.equal(result.total_events, 2)
    assert.equal(result.events[0].event_type, 'PLANNER_COMPLETED')
  })

  test('fetchRunLogs safely returns empty structure when runId is empty', async () => {
    const result = await fetchRunLogs('')
    assert.equal(result.total_events, 0)
    assert.deepEqual(result.events, [])
  })
})

describe('Stage C: Base-Paper Citation Integrity', () => {
  test('Correct base paper citation and DOI are referenced', () => {
    const citation = {
      title: 'AutoMind: Automated Insight Discovery via Multi-Agent Navigation',
      journal: 'IEEE Access',
      year: 2026,
      volume: 14,
      pages: '23936–23955',
      doi: '10.1109/ACCESS.2026.3661202',
      ieeeXplore: 'https://ieeexplore.ieee.org/document/11372738/',
    }

    assert.equal(citation.title, 'AutoMind: Automated Insight Discovery via Multi-Agent Navigation')
    assert.equal(citation.journal, 'IEEE Access')
    assert.equal(citation.doi, '10.1109/ACCESS.2026.3661202')
    assert.equal(citation.volume, 14)
    assert.match(citation.ieeeXplore, /11372738/)
  })

  test('fetchClassificationBenchmark fetches /api/benchmarks/classification successfully', async () => {
    let capturedUrl = null
    globalThis.fetch = async (url) => {
      capturedUrl = url
      return {
        ok: true,
        json: async () => ({
          status: 'success',
          summary_records: [
            { dataset: 'Heart Disease (Staged 918-row)', model_type: 'logistic_regression', baseline_f1: 0.9005, proposed_f1: 0.9038 }
          ],
          automind_comparison: [
            { "Framework / System": "AutoMind (Published IEEE Access 2026 Table 3)", "F1-Score (%)": "95.52%" }
          ]
        }),
      }
    }

    const data = await fetchClassificationBenchmark()
    assert.equal(capturedUrl, '/api/benchmarks/classification')
    assert.equal(data.status, 'success')
    assert.equal(data.automind_comparison[0]['F1-Score (%)'], '95.52%')
    assert.equal(data.summary_records[0].model_type, 'logistic_regression')
  })
})
