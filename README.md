# DataInvest AI 📊

> **Automated Insight Discovery & Statistical Investigation** — inspired by the AutoMind research foundation, delivering deterministic dataset profiling, quality diagnostics, and structured analysis task planning.

---

## System Architecture

DataInvest AI is built with a modular full-stack architecture:
- **Backend:** Python + FastAPI + Pandas + NumPy
- **Frontend:** React + Vite with a modern dark navy (`#071525`) & cyan (`#18C8FF`) visual system
- **Statistical Engine:** Pure deterministic calculations using Pandas/NumPy (no hallucinations, no fabricated statistics)
- **Planning Engine:** Phase 2 Planner Agent that translates natural-language goals into structured, prioritized investigation tasks

---

## Phase Breakdown

```
CSV Upload
    ↓
Dataset Profiler (Phase 1)
    ↓
User Investigation Objective
    ↓
Planner Agent (Phase 2)
    ↓
Structured Investigation Tasks
    ↓
[Phase 3: Analysis Agent - Coming Next]
```

- **Phase 1 (Completed & Validated):** CSV Upload & Deterministic Dataset Profiler
- **Phase 2 (Completed & Validated):** Analysis Planner Agent (Goal Decomposition & Task Planning)
- **Phase 3:** Analysis Agent (Statistical Execution with SciPy/Statsmodels)
- **Phase 4:** Critic Agent (Quality Assurance & Hypothesis Validation)
- **Phase 5:** Report Agent (Automated Research Report Generation)

---

## Project Structure

```
Major_Project/
├── backend/
│   ├── app/
│   │   ├── main.py                  ← FastAPI entry point & routes (/api/profile, /api/plan)
│   │   ├── agents/
│   │   │   └── planner_agent.py     ← Phase 2 Planner Agent logic
│   │   ├── modules/
│   │   │   └── profiler.py          ← Phase 1 Dataset profiler engine
│   │   ├── schemas/
│   │   │   ├── profile.py           ← Pydantic profile response models
│   │   │   └── planner_schema.py    ← Pydantic planner task & plan models
│   │   └── services/
│   │       ├── upload_service.py    ← CSV parsing & validation
│   │       └── planner_service.py   ← Planning orchestration service
│   ├── tests/
│   │   ├── test_api.py              ← Phase 1 & 2 FastAPI integration tests
│   │   ├── test_profiler.py         ← Phase 1 profiler unit tests
│   │   └── test_planner.py          ← Phase 2 planner agent unit tests
│   ├── conftest.py
│   └── requirements.txt
│
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── Navbar.jsx           ← Top navigation bar
│   │   │   ├── Hero.jsx             ← Minimal hero section
│   │   │   ├── UploadCard.jsx       ← Drag-and-drop CSV upload card
│   │   │   ├── MetricCard.jsx       ← Key metrics (rows, cols, missing, dups)
│   │   │   ├── DatasetPreview.jsx   ← Interactive first-N rows table
│   │   │   ├── ColumnProfile.jsx    ← Column types, missing %, unique counts
│   │   │   ├── NumericalProfile.jsx ← Mean, median, std, quartiles, IQR, outliers
│   │   │   ├── DataQuality.jsx      ← Data quality checklist with status badges
│   │   │   ├── PlannerSection.jsx   ← Phase 2 Objective input & Task plan cards
│   │   │   ├── LoadingState.jsx     ← Modern spinner state
│   │   │   └── ErrorState.jsx       ← User-friendly error display
│   │   ├── pages/
│   │   │   └── Home.jsx             ← Dashboard orchestrator
│   │   ├── services/
│   │   │   └── api.js               ← API client for /api/profile & /api/plan
│   │   ├── App.jsx                  ← Application root state machine
│   │   ├── main.jsx                 ← React entry point
│   │   └── index.css                ← Strict color system & global styles
│   ├── package.json
│   └── vite.config.js               ← Vite config with /api proxy
│
├── docs/
│   └── PHASE_2_PLANNER.md           ← Technical planner specification
├── data/
│   └── sample.csv                   ← Sample employee dataset
├── requirements.txt
└── README.md
```

---

## Installation & Setup

### Prerequisites
- **Python 3.10+**
- **Node.js 18+** & **npm**

### 1. Backend Setup

```bash
# From project root
pip install -r requirements.txt

# Start FastAPI backend server
cd backend
python -m uvicorn app.main:app --reload --port 8000
```
- API Docs (Swagger): `http://127.0.0.1:8000/docs`
- Health check: `http://127.0.0.1:8000/health`

### 2. Frontend Setup

```bash
# In a separate terminal
cd frontend
npm install
npm run dev
```
- Web Application: `http://127.0.0.1:5173/`

---

## Running Test Suite

```bash
cd backend
python -m pytest tests/ -v
```

All 82 backend tests validate:
- **Phase 1**: Dimensions, missing-value percentages, duplicate detection, exact numerical stats, IQR outlier bounds, constant/high-cardinality detection, encoding resilience, and HTTP endpoints.
- **Phase 2**: Objective interpretation, target variable detection, non-symmetric bivariate planning (Pearson/Spearman), group comparisons (Welch's t-test / ANOVA / Mann-Whitney / Kruskal-Wallis), contingency tests (Chi-square / Cramér's V), outlier sensitivity tasks, high-cardinality exclusions, Pydantic contracts, and `POST /api/plan` integration.
