// constants/phases.js — Central Investigation Pipeline Phase Definitions

export const PHASES = [
  {
    id: 1,
    key: 'profile',
    label: 'Profile',
    shortLabel: '01 Profile',
    title: '01 · Dataset Profile',
    badge: 'Phase 1 · Dataset Profiling',
    stepNum: '01',
    description: 'Structure, missingness, cardinality, distributions & data quality audit',
  },
  {
    id: 2,
    key: 'plan',
    label: 'Plan',
    shortLabel: '02 Plan',
    title: '02 · Investigation Plan',
    badge: 'Phase 2 · Investigation Planner',
    stepNum: '02',
    description: 'Target identification, variable mapping & prioritized task planning',
  },
  {
    id: 3,
    key: 'analyze',
    label: 'Analyze',
    shortLabel: '03 Analyze',
    title: '03 · Statistical Analysis',
    badge: 'Phase 3 · Statistical Execution',
    stepNum: '03',
    description: 'SciPy bivariate tests, group comparisons, ANOVA & outlier sensitivity',
  },
  {
    id: 4,
    key: 'critic',
    label: 'Critic',
    shortLabel: '04 Critic',
    title: '04 · Critic Verification',
    badge: 'Phase 4 · Critic Agent Verification',
    stepNum: '04',
    description: 'Methodological checks, sample adequacy, outliers & assumption audits',
  },
  {
    id: 5,
    key: 'verify',
    label: 'Verify',
    shortLabel: '05 Verify',
    title: '05 · Verified Findings',
    badge: 'Phase 5 · Re-investigation & Verified Findings',
    stepNum: '05',
    description: 'Targeted re-testing, before/after deltas & synthesized verified findings',
  },
  {
    id: 6,
    key: 'insights',
    label: 'Insights',
    shortLabel: '06 Insights',
    title: '06 · Key Insights',
    badge: 'Phase 6 · Verified Insight Generation',
    stepNum: '06',
    description: 'Human-readable analytical insights derived exclusively from verified statistical evidence',
  },
  {
    id: 7,
    key: 'evaluation',
    label: 'Evaluation',
    shortLabel: '07 Evaluation',
    title: '07 · System Evaluation & Benchmark Scorecard',
    badge: 'Phase 7 · System Evaluation',
    stepNum: '07',
    description: 'Quantitative benchmark scorecard evaluating pipeline reliability, statistical rigor & empirical fidelity',
  },
]


export function getPhaseById(id) {
  return PHASES.find((p) => p.id === id) || PHASES[0]
}

export function getPhaseByKey(key) {
  return PHASES.find((p) => p.key === key) || PHASES[0]
}
