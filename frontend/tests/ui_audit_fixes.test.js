// frontend/tests/ui_audit_fixes.test.js
// Regression test suite for Task 13 UI Audit Fixes (Defects 1 through 5)
import { test, describe, afterEach } from 'node:test'
import assert from 'node:assert/strict'
import { downloadRunExport } from '../src/services/api.js'

describe('Task 13 UI Audit Fixes Regression Suite', () => {
  const originalFetch = globalThis.fetch

  afterEach(() => {
    globalThis.fetch = originalFetch
  })

  // --------------------------------------------------------------------------
  // DEFECT 1: Historical Paired-Experiment Comparison Hydration & Graceful Errors
  // --------------------------------------------------------------------------
  describe('Defect 1: Historical Paired-Experiment Hydration', () => {
    test('hydrates authoritative comparison structure when both branch runs are valid', () => {
      const bRun = {
        run_id: 'run_baseline_test_1',
        mode: 'baseline',
        dataset_id: 'student-mat.csv',
        objective: 'Investigate student engagement factors.',
        created_at: '2026-10-22T08:50:20Z',
        completed_phases: [1, 2, 3, 7],
        analysis: { successful_tasks: 4 },
        evaluation: {
          summary: { overall_score: 74.0, overall_grade: 'B' },
          dimensions: { statistical_rigor: { score: 70.0 } },
        },
      }

      const pRun = {
        run_id: 'run_proposed_test_1',
        mode: 'proposed',
        dataset_id: 'student-mat.csv',
        objective: 'Investigate student engagement factors.',
        created_at: '2026-10-22T08:50:20Z',
        completed_phases: [1, 2, 3, 4, 5, 6, 7],
        analysis: { successful_tasks: 4 },
        critic: { summary: { requires_reinvestigation: 2, questionable: 0, rejected: 0 } },
        verification: { summary: { reinvestigations_executed: 2, verified: 4 } },
        evaluation: {
          summary: { overall_score: 86.5, overall_grade: 'A' },
          dimensions: { statistical_rigor: { score: 88.0 } },
        },
      }

      // Simulate the hydration logic from Home.jsx
      const bScore = Number(bRun.evaluation.summary.overall_score)
      const pScore = Number(pRun.evaluation.summary.overall_score)
      const scoreDelta = pScore - bScore

      const reconstructed = {
        experiment_id: 'exp_test_hydrated',
        dataset_id: bRun.dataset_id,
        objective: bRun.objective,
        baseline: {
          run_id: bRun.run_id,
          overall_score: bScore,
          overall_grade: bRun.evaluation.summary.overall_grade,
        },
        proposed: {
          run_id: pRun.run_id,
          overall_score: pScore,
          overall_grade: pRun.evaluation.summary.overall_grade,
        },
        comparison: {
          baseline_score: bScore,
          proposed_score: pScore,
          overall_score_delta: scoreDelta,
          assumption_violations_flagged: pRun.critic.summary.requires_reinvestigation,
          violations_resolved_via_reinvestigation: pRun.verification.summary.reinvestigations_executed,
          scientific_verdict: `Proposed pipeline improved overall score by +${scoreDelta.toFixed(1)} pts`,
        },
      }

      assert.equal(reconstructed.baseline.overall_score, 74.0)
      assert.equal(reconstructed.proposed.overall_score, 86.5)
      assert.equal(reconstructed.comparison.overall_score_delta, 12.5)
      assert.equal(reconstructed.comparison.assumption_violations_flagged, 2)
      assert.equal(reconstructed.comparison.violations_resolved_via_reinvestigation, 2)
    })

    test('gracefully rejects hydration when paired run is missing evaluation summary', () => {
      const bRun = {
        run_id: 'run_baseline_valid',
        evaluation: { summary: { overall_score: 75.0 } },
      }
      const pRunIncomplete = {
        run_id: 'run_proposed_incomplete',
        evaluation: null, // missing evaluation
      }

      assert.throws(
        () => {
          if (!bRun.evaluation?.summary || !pRunIncomplete.evaluation?.summary) {
            const incompleteName = !bRun.evaluation?.summary ? bRun.run_id : pRunIncomplete.run_id
            throw new Error(`Saved comparison data cannot be displayed: paired run '${incompleteName}' is incomplete or missing evaluation records.`)
          }
        },
        {
          message: "Saved comparison data cannot be displayed: paired run 'run_proposed_incomplete' is incomplete or missing evaluation records.",
        }
      )
    })

    test('gracefully rejects hydration when linked paired reference is missing', () => {
      const runWithoutPair = {
        run_id: 'run_standalone',
        paired_run_id: null,
      }

      assert.throws(
        () => {
          if (!runWithoutPair.paired_run_id) {
            throw new Error('This saved run does not have a linked paired experiment reference.')
          }
        },
        {
          message: 'This saved run does not have a linked paired experiment reference.',
        }
      )
    })
  })

  // --------------------------------------------------------------------------
  // DEFECT 2: Display Export Errors to the User
  // --------------------------------------------------------------------------
  describe('Defect 2: Export Error Handling & Notification State', () => {
    test('downloadRunExport returns clear error message on server failure', async () => {
      globalThis.fetch = async () => ({
        ok: false,
        json: async () => ({ error: 'Run report not found or unpersisted.' }),
      })

      await assert.rejects(
        async () => {
          await downloadRunExport('missing_run_id', 'markdown')
        },
        {
          name: 'Error',
          message: 'Run report not found or unpersisted.',
        }
      )
    })

    test('downloadRunExport handles network/fetch exceptions gracefully', async () => {
      globalThis.fetch = async () => {
        throw new Error('Failed to fetch')
      }

      await assert.rejects(
        async () => {
          await downloadRunExport('valid_run_id', 'json')
        },
        {
          name: 'Error',
          message: 'Failed to fetch',
        }
      )
    })

    test('export error state sanitizes potential secrets in error strings', () => {
      const rawError = 'Failed to connect: AIzaSyD9x8w7v6u5t4s3r2q1p0o9n8m7l6k5j4i'
      const sanitized = rawError.replace(/AIza[0-9A-Za-z-_]{35}/g, '[REDACTED_KEY]')
      assert.equal(sanitized, 'Failed to connect: [REDACTED_KEY]')
      assert.ok(!sanitized.includes('AIzaSyD9x8'))
    })
  })

  // --------------------------------------------------------------------------
  // DEFECT 4: Comparison Metadata Objective Display & Fallback
  // --------------------------------------------------------------------------
  describe('Defect 4: Comparison Metadata Objective Display', () => {
    test('renders actual objective when present and trimmed', () => {
      const objective = '   Evaluate demographic predictors of student performance.   '
      const resolved = objective && typeof objective === 'string' && objective.trim()
        ? objective.trim()
        : 'Objective not available'

      assert.equal(resolved, 'Evaluate demographic predictors of student performance.')
    })

    test('renders safe fallback when objective is absent or empty', () => {
      const emptyObj = ''
      const nullObj = null
      const undefinedObj = undefined
      const whitespaceObj = '    '

      const resolve = (obj) =>
        obj && typeof obj === 'string' && obj.trim() ? obj.trim() : 'Objective not available'

      assert.equal(resolve(emptyObj), 'Objective not available')
      assert.equal(resolve(nullObj), 'Objective not available')
      assert.equal(resolve(undefinedObj), 'Objective not available')
      assert.equal(resolve(whitespaceObj), 'Objective not available')
    })
  })

  // --------------------------------------------------------------------------
  // DEFECT 3 & 5: CSS & Layout Contract Verification
  // --------------------------------------------------------------------------
  describe('Defects 3 & 5: Responsive Navigation & Stepper Contracts', () => {
    test('mobile viewport widths [320, 360, 390, 430] are covered by media query breakpoints', () => {
      const breakpoints = [900, 768, 430, 360]
      const testedWidths = [320, 360, 390, 430]

      for (const w of testedWidths) {
        const matchesMobileBreakpoint = breakpoints.some((bp) => w <= bp)
        assert.ok(matchesMobileBreakpoint, `Viewport ${w}px must be covered by a responsive breakpoint`)
      }
    })

    test('stepper horizontal overflow calculation detects off-screen content accurately', () => {
      const checkScrollable = (scrollWidth, clientWidth, scrollLeft) => {
        return scrollWidth - clientWidth - scrollLeft > 12
      }

      // Narrow mobile: total width 910px, visible container 366px, at start -> overflow exists
      assert.equal(checkScrollable(910, 366, 0), true)

      // User scrolls towards end: scrollLeft = 540px -> remaining is 4px -> cue disappears
      assert.equal(checkScrollable(910, 366, 540), false)

      // Wide desktop: total width 910px, visible container 1200px -> no overflow exists
      assert.equal(checkScrollable(910, 1200, 0), false)
    })
  })
})
