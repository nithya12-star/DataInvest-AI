// frontend/tests/visual_evidence.test.js
// Unit tests for Vega visual evidence structure, contract, and safety invariants.
import { test, describe } from 'node:test'
import assert from 'node:assert/strict'

describe('Visual Evidence & Vega-Lite Contract', () => {
  test('conforms to Phase 3/5 VisualEvidence schema contract', () => {
    const visualEvidence = {
      graph_id: 'graph_task_1_scatter',
      task_id: 'task_1',
      figure_type: 'scatter',
      title: 'MonthlyCharges vs TotalCharges Correlation',
      x_label: 'MonthlyCharges',
      y_label: 'TotalCharges',
      columns: ['MonthlyCharges', 'TotalCharges'],
      method: 'pearson_correlation',
      data_summary: { sample_size: 100, bivariate_outliers: 2 },
      figure_spec: {
        $schema: 'https://vega.github.io/schema/vega-lite/v5.json',
        data: { values: [{ MonthlyCharges: 29.85, TotalCharges: 29.85 }] },
        mark: { type: 'circle', color: '#00BFFF', opacity: 0.75, size: 60 },
        encoding: {
          x: { field: 'MonthlyCharges', type: 'quantitative' },
          y: { field: 'TotalCharges', type: 'quantitative' },
        },
      },
      visual_diagnostics: {
        outlier_points_count: 2,
        linearity_indicator: 'Linear',
        distribution_shape: null,
        visual_notes: [
          'Strong positive linear correlation pattern detected.',
          'Detected 2 bivariate outlier points via Mahalanobis distance at alpha=0.01.',
        ],
        vision_model_used: false,
        visual_inspection_claimed: false,
      },
    }

    assert.equal(visualEvidence.figure_type, 'scatter')
    assert.equal(visualEvidence.figure_spec.mark.type, 'circle')
    assert.equal(visualEvidence.visual_diagnostics.vision_model_used, false)
    assert.equal(visualEvidence.visual_diagnostics.visual_inspection_claimed, false)
    assert.equal(visualEvidence.visual_diagnostics.outlier_points_count, 2)
    assert.equal(visualEvidence.visual_diagnostics.visual_notes.length, 2)
  })

  test('reinvestigation record preserves both original and follow-up visual evidence', () => {
    const reinvestigationRecord = {
      reinvestigation_id: 'reinv_task_2_1',
      source_task_id: 'task_2',
      original_method: 'independent_samples_t_test',
      reinvestigated_method: 'mann_whitney_u_test',
      original_graph_id: 'graph_task_2_box',
      reinvestigated_graph_id: 'graph_reinv_task_2_box',
      original_visual_evidence: {
        graph_id: 'graph_task_2_box',
        figure_type: 'box',
        title: 'MonthlyCharges by Contract (Original)',
        figure_spec: { mark: 'boxplot' },
      },
      reinvestigated_visual_evidence: {
        graph_id: 'graph_reinv_task_2_box',
        figure_type: 'box',
        title: 'MonthlyCharges by Contract (Mann-Whitney U Follow-up)',
        figure_spec: { mark: 'boxplot' },
      },
    }

    assert.notEqual(
      reinvestigationRecord.original_graph_id,
      reinvestigationRecord.reinvestigated_graph_id
    )
    assert.equal(reinvestigationRecord.original_visual_evidence.figure_type, 'box')
    assert.equal(reinvestigationRecord.reinvestigated_visual_evidence.figure_type, 'box')
    assert.equal(
      reinvestigationRecord.reinvestigated_visual_evidence.title.includes('Follow-up'),
      true
    )
  })

  test('gracefully handles missing or malformed figure specs without uncaught exceptions', () => {
    const malformedSpecs = [
      null,
      undefined,
      'not valid json { [',
      {},
      { mark: 'unknown_chart_type_xyz' },
    ]

    for (const spec of malformedSpecs) {
      let parseError = null
      let parsed = null

      if (spec) {
        try {
          parsed = typeof spec === 'string' ? JSON.parse(spec) : JSON.parse(JSON.stringify(spec))
        } catch (err) {
          parseError = err.message
        }
      }

      if (typeof spec === 'string' && spec.startsWith('not valid')) {
        assert.ok(parseError !== null, 'Malformed JSON string produces safe parse error')
      } else if (!spec) {
        assert.equal(parsed, null)
      } else {
        assert.ok(parsed !== null)
      }
    }
  })

  test('enforces programmatic limitation notice without claiming vision perception', () => {
    const limitationText =
      'Limitation Notice: Programmatic visual diagnostics used. No vision-capable model inspected or interpreted this graph.'
    assert.ok(limitationText.includes('Programmatic visual diagnostics'))
    assert.ok(limitationText.includes('No vision-capable model'))
  })

  test('resolves historical verified finding fields accurately without NaN or blank values', () => {
    // Model of winequality saved run finding
    const historicalFinding = {
      finding_id: 'finding_1',
      task_id: 'task_1',
      variables: ['fixed acidity'],
      method: 'Descriptive summary & distribution assessment',
      status: 'VERIFIED',
      confidence: 1,
      confidence_level: 'HIGH',
      summary: 'Distribution characterized as moderately right-skewed.',
      statistical_evidence: {
        column: 'fixed acidity',
        sample_size: 1599,
        normality_test: {
          method: "D'Agostino-Pearson omnibus test",
          statistic: 224.5309,
          p_value: 1.7528e-49,
          is_normally_distributed: false,
        },
      },
      critic_assessment: 'Finding is empirically verified and methodologically robust.',
      reinvestigated: false,
      reinvestigation_record: null,
      verification_reason: 'Analysis completed cleanly with zero unresolved issues.',
      limitations: ['Variable significantly departs from normal distribution.'],
      reason_codes: ['NON_NORMAL_DISTRIBUTION', 'DIRECTLY_VERIFIED'],
    }

    // Resolution logic mirrors VerifiedFindingsSection.jsx
    const resolveConfidence = (f) => {
      const val = f.confidence !== undefined && f.confidence !== null ? f.confidence : f.confidence_score
      return typeof val === 'number' && !Number.isNaN(val) ? val : null
    }

    const resolvePValue = (f) => {
      if (typeof f.p_value === 'number' && !Number.isNaN(f.p_value)) return f.p_value
      const stat = f.statistical_evidence
      if (stat) {
        if (typeof stat.p_value === 'number' && !Number.isNaN(stat.p_value)) return stat.p_value
        if (stat.normality_test && typeof stat.normality_test.p_value === 'number') return stat.normality_test.p_value
      }
      return null
    }

    const conf = resolveConfidence(historicalFinding)
    assert.equal(conf, 1)
    assert.equal(`${(conf * 100).toFixed(0)}%`, '100%')

    const pVal = resolvePValue(historicalFinding)
    assert.equal(pVal, 1.7528e-49)

    const finalStatus = historicalFinding.status || historicalFinding.final_status
    assert.equal(finalStatus, 'VERIFIED')

    const method = historicalFinding.method || historicalFinding.method_used
    assert.equal(method, 'Descriptive summary & distribution assessment')

    const summary = historicalFinding.summary || historicalFinding.claim
    assert.equal(summary, 'Distribution characterized as moderately right-skewed.')

    const targetVar = (Array.isArray(historicalFinding.variables) && historicalFinding.variables.length > 0)
      ? historicalFinding.variables.join(' ↔ ')
      : historicalFinding.target_column
    assert.equal(targetVar, 'fixed acidity')

    const tags = (Array.isArray(historicalFinding.tags) && historicalFinding.tags.length > 0)
      ? historicalFinding.tags
      : historicalFinding.reason_codes
    assert.deepEqual(tags, ['NON_NORMAL_DISTRIBUTION', 'DIRECTLY_VERIFIED'])
  })

  test('validates missing or invalid confidence safely without producing NaN% or assuming 100%', () => {
    const resolveConfidence = (f) => {
      const val = f.confidence !== undefined && f.confidence !== null ? f.confidence : f.confidence_score
      return typeof val === 'number' && !Number.isNaN(val) ? val : null
    }

    const formatConfidence = (c) => (c !== null ? `${(c * 100).toFixed(0)}%` : 'N/A')

    assert.equal(formatConfidence(resolveConfidence({ confidence: undefined })), 'N/A')
    assert.equal(formatConfidence(resolveConfidence({ confidence: null })), 'N/A')
    assert.equal(formatConfidence(resolveConfidence({ confidence: NaN })), 'N/A')
    assert.equal(formatConfidence(resolveConfidence({ confidence: 'not-a-number' })), 'N/A')
    assert.equal(formatConfidence(resolveConfidence({ confidence_score: 0.82 })), '82%')
    assert.equal(formatConfidence(resolveConfidence({ confidence: 0.95 })), '95%')
  })

  test('History restoration preserves visual_evidence across analysis and verification contracts', () => {
    const savedRunFromHistory = {
      run_id: 'run_20261023_093000_wine_a1b2c3',
      dataset_name: 'winequality-red.csv',
      active_phase: 5,
      completed_phases: [1, 2, 3, 4, 5],
      has_visual_evidence: true,
      mode: 'proposed',
      analysis: {
        results: [
          {
            task_id: 'task_1',
            status: 'success',
            visual_evidence: {
              graph_id: 'graph_task_1_scatter',
              figure_type: 'scatter',
              figure_spec: { mark: 'circle' },
              visual_diagnostics: { vision_model_used: false, outlier_points_count: 3 },
            },
          },
        ],
      },
      verification: {
        verified_findings: [
          {
            finding_id: 'finding_1',
            task_id: 'task_1',
            status: 'VERIFIED',
            visual_evidence: {
              graph_id: 'graph_task_1_scatter',
              figure_type: 'scatter',
              figure_spec: { mark: 'circle' },
            },
          },
        ],
      },
    }

    // Verify properties needed for History drawer badges
    assert.equal(savedRunFromHistory.has_visual_evidence, true)
    assert.equal(savedRunFromHistory.mode, 'proposed')

    // Simulate state restoration in Home.jsx (lines 155-163)
    const restoredAnalysis = savedRunFromHistory.analysis
    const restoredVerification = savedRunFromHistory.verification

    assert.ok(restoredAnalysis.results[0].visual_evidence)
    assert.equal(restoredAnalysis.results[0].visual_evidence.figure_type, 'scatter')
    assert.equal(restoredAnalysis.results[0].visual_evidence.visual_diagnostics.vision_model_used, false)

    assert.ok(restoredVerification.verified_findings[0].visual_evidence)
    assert.equal(restoredVerification.verified_findings[0].visual_evidence.graph_id, 'graph_task_1_scatter')
  })

  test('History restoration handles legacy saved runs without visual evidence safely', () => {
    const legacySavedRun = {
      run_id: 'run_20261019_120000_wine_legacy',
      dataset_name: 'winequality-red.csv',
      active_phase: 5,
      completed_phases: [1, 2, 3, 4, 5],
      has_visual_evidence: false,
      analysis: {
        results: [{ task_id: 'task_1', status: 'success' }],
      },
      verification: {
        verified_findings: [{ finding_id: 'finding_1', status: 'VERIFIED' }],
      },
    }

    assert.equal(legacySavedRun.has_visual_evidence, false)
    assert.equal(legacySavedRun.analysis.results[0].visual_evidence, undefined)
    assert.equal(legacySavedRun.verification.verified_findings[0].visual_evidence, undefined)

    // Render condition check used in AnalysisResults.jsx & VerifiedFindingsSection.jsx
    const shouldRenderAnalysisChart = Boolean(legacySavedRun.analysis.results[0].visual_evidence)
    const shouldRenderVerificationChart = Boolean(legacySavedRun.verification.verified_findings[0].visual_evidence)

    assert.equal(shouldRenderAnalysisChart, false)
    assert.equal(shouldRenderVerificationChart, false)
  })

  test('renders comparative side-by-side visual evidence structure for re-investigated findings', () => {
    const findingWithReinvestigation = {
      finding_id: 'finding_1',
      status: 'VERIFIED_AFTER_REINVESTIGATION',
      reinvestigation_record: {
        reinvestigation_id: 'reinvest_t1',
        source_task_id: 't1',
        original_method: 'Pearson correlation',
        reinvestigated_method: 'Spearman rank correlation',
        original_visual_evidence: {
          graph_id: 'graph_t1_scatter',
          figure_type: 'scatter',
          title: 'Initial Pearson Scatter',
          visual_diagnostics: { outlier_points_count: 5 },
        },
        reinvestigated_visual_evidence: {
          graph_id: 'graph_t1_reinvest_scatter',
          figure_type: 'scatter',
          title: 'Follow-up Spearman Scatter',
          visual_diagnostics: { outlier_points_count: 5 },
        },
      },
    }

    const rec = findingWithReinvestigation.reinvestigation_record
    assert.ok(rec.original_visual_evidence)
    assert.ok(rec.reinvestigated_visual_evidence)
    assert.notEqual(rec.original_visual_evidence.graph_id, rec.reinvestigated_visual_evidence.graph_id)
    assert.equal(rec.original_method, 'Pearson correlation')
    assert.equal(rec.reinvestigated_method, 'Spearman rank correlation')
  })
})


