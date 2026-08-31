# Model 2: Project Execution Delay & Stalling Engine — In-Depth Architecture & Training Specification

> **Model Identifier:** Model 2 (Pillar 2: Project Execution Delay & Governance Engine)  
> **Model Architecture:** **Hybrid Multi-Stage Latency Scorer & Multivariate Isolation Forest (`sklearn.ensemble.IsolationForest`)**  
> **Source Files:** [`ExecutionModelModule.py`](file:///c:/Users/DIVYA/OneDrive/Desktop/3_Projects_and_Development/Projects/SIH/mplads-risk-analytics/ExecutionModelModule.py), [`train_execution_model.py`](file:///c:/Users/DIVYA/OneDrive/Desktop/3_Projects_and_Development/Projects/SIH/mplads-risk-analytics/train_execution_model.py), [`app/services/execution_feature_engineering.py`](file:///c:/Users/DIVYA/OneDrive/Desktop/3_Projects_and_Development/Projects/SIH/mplads-risk-analytics/app/services/execution_feature_engineering.py)  
> **Artifact Path:** `models/execution_delay_model.joblib`

---

## 1. Machine Learning Model Architecture & Rationale

### A. Model Choice: Isolation Forest (`IsolationForest`)
We chose an **Isolation Forest** ensemble algorithm combined with `StandardScaler` and a **Domain Rule Engine**:

- **Model Parameters:**
  - `n_estimators = 300` (Ensemble of 300 decision trees)
  - `contamination = 0.05` (5% expected anomaly proportion baseline)
  - `random_state = 42` (Reproducible split & scoring)
  - `n_jobs = -1` (Vectorized multi-core parallel processing)

### B. Why We Chose Isolation Forest
1. **Unsupervised Anomaly Isolation:** Government MPLADS data lacks labeled ground-truth binary targets (e.g. no explicit `"corrupt"` or `"stalled"` flags in raw portal exports). Isolation Forest isolates rare execution anomalies—such as severe project age spikes, unusual cost escalation ratios, or extreme approval latencies—without needing labeled historical training targets.
2. **High-Dimensional Latency Modeling:** Civil projects progress through multiple asynchronous dates (`recommended_date`, `sanction_date`, `completion_date`). Isolation Forest models non-linear interactions across these temporal dimensions simultaneously.
3. **Sub-Millisecond Inference Speed:** Single-row prediction runs in `< 1ms`, allowing real-time batch evaluations across thousands of constituency work orders during API requests.

### C. Why We Use a Hybrid ML + Rule Engine Approach
Pure machine learning models can operate as opaque "black boxes." In government oversight applications, district collectors and vigilance officers require **legally actionable, explainable evidence**. 

Our hybrid architecture maps Isolation Forest vector outputs to statutory MPLADS guidelines:
- **ML Isolation Forest:** Detects multivariate statistical outliers in execution lead times, cost ratios, and MP delivery track records.
- **Governance Rule Engine:** Applies statutory rules (e.g., 180-day completion threshold, 365-day chronic stalling limit, mandatory geo-tagged photo evidence) to produce human-understandable audit warnings (`flagged_reasons`) and explainability chips (`explainability_tags`).

---

## 2. Dataset Preprocessing & Feature Engineering

### A. Raw Datasets Used
The model trains and builds lookups from 4 datasets located in `New Datasets/`:
1. `Works Recommended.csv`: Initial proposals by MPs.
2. `Works Sanctioned.csv`: Administrative approvals by Implementing District Authorities (IDAs).
3. `Works Completed.csv`: Physical completion logs, photo upload indicators, and final expenditure.
4. `Expenditure on Completed and On-going Works as on Date.csv`: Disbursement records.

### B. In-Memory Baseline Index Building (`build_execution_indexes`)
On application boot or training, the engine constructs three fast lookup indexes:
- **`mp_completion_index`:** Tracks each MP's historical completion rate (`Completed Works / Sanctioned Works`).
- **`state_latency_index`:** Calculates historical baseline recommendation-to-sanction approval latency per State/UT (e.g. Karnataka avg = 28.4 days).
- **`ida_latency_index`:** Calculates baseline approval latency per Implementing District Authority (IDA).

### C. Engineered Feature Vector (9 Dimensional Inputs)

| # | Feature Name | Formula / Logic | Purpose & Significance |
| :-: | :--- | :--- | :--- |
| 1 | `approval_latency_days` | `Sanction Date - Recommended Date` | Measures administrative bureaucratic delay. |
| 2 | `project_age_days` | `Completion Date - Sanction Date` (if completed) or `Today - Sanction Date` (if ongoing) | Measures physical execution duration or active stalled age. |
| 3 | `total_lead_time_days` | `Completion Date - Recommended Date` | Measures full end-to-end project turnaround. |
| 4 | `is_stalled_365` | `1` if ongoing project age > 365 days, else `0` | Binary indicator for chronic zombie projects. |
| 5 | `missing_photo_penalty` | `40.0` if `has_photo_evidence == False`, else `0.0` | Heavy penalty for unverified public fund disbursement. |
| 6 | `status_risk_factor` | Stage weight: `Physical Inspection` (0.8), `Vendor Tender` (0.7), `Ongoing` (0.5), `Sanction` (0.4) | Quantifies milestone chokepoint severity. |
| 7 | `cost_escalation_ratio` | `amount_disbursed / sanction_amount` | Detects budget inflation or over-disbursement (> 1.05). |
| 8 | `mp_completion_ratio` | `MP Completed / Total Sanctioned` | Contextual track record of constituency execution efficiency. |
| 9 | `approval_latency_deviation` | `approval_latency_days - state_baseline_latency` | Measures deviation from regional administrative norms. |

---

## 3. Training & Persistence Pipeline (`train_execution_model.py`)

### Step-by-Step Training Process:
```
[Raw CSVs] ──> [Date Standardization] ──> [Build MP/State Baselines] ──> [Feature Vector Generation] ──> [StandardScaler] ──> [Isolation Forest Fit] ──> [Export .joblib]
```

1. **Date Parsing & Standardization:** Standardizes mixed date string formats (`08-Jul-2024`, `2024-07-08`, `05/09/2024`) using robust date coercion.
2. **Constructing Training Samples:** Joins sanctioned and completed work records into unified milestone objects.
3. **Feature Scaling:** Fits `StandardScaler()` on the 9-dimensional feature matrix `X` to standardize zero mean and unit variance.
4. **Model Fitting:** Fits `IsolationForest(n_estimators=300, contamination=0.05)` on scaled matrix `X_scaled`.
5. **Decision Score Calibration:** Computes raw decision function scores (`self.model.decision_function`), recording `_score_min` and `_score_max` bounds for min-max normalization.
6. **Artifact Export:** Saves the trained pipeline to `models/execution_delay_model.joblib` using `joblib.dump`:
   - `model`: Trained Isolation Forest instance
   - `scaler`: Fitted `StandardScaler`
   - `feature_cols`: List of feature names
   - `_score_min` & `_score_max`: Normalization bounds
   - `background`: Sample matrix (100 rows) saved for SHAP explainability calculations

---

## 4. Scoring, Risk Level Categorization, & Decision Logic

When evaluating a work order, the engine computes a composite **Execution Risk Score** (0.0 to 100.0) combining rule penalties and ML decision outputs:

### A. Scoring & Penalty Rules
1. **Missing Photo Evidence (+40.0 pts):** Added if funds are disbursed without geo-tagged mobile application photographic evidence (`has_photo_evidence == False`).
2. **Chronic Stalling >365 Days (+45.0 pts):** Added if project has been active without completion for over 365 days.
3. **Stage Delay >180 Days (+20.0 pts):** Added if active execution surpasses the standard 180-day delivery threshold.
4. **Stage Chokepoints:** Adds specific chokepoint tags (`PHYSICAL_INSPECTION_BOTTLENECK` if stalled at physical inspection > 90d, `VENDOR_TENDER_BOTTLENECK` if stalled at tender stage > 60d).
5. **Blocked Public Capital:** Flags projects where ≥ ₹2.0 Lakh is disbursed without Final Completion Certificate (FCC).
6. **Cost Escalation (+15.0 max pts):** Adds points proportional to percentage budget overrun if `disbursed / sanction > 1.05`.
7. **On-Time Reward (Cap = 15.0 pts):** Completed projects with verified photos, within budget, and completed ≤ 240 days are capped at a low compliant score.

### B. Risk Level Categorization & Recommended Governance Actions

| Execution Risk Score | Risk Level | Compliance Status | Governance Action Triggered |
| :---: | :---: | :---: | :--- |
| **70.0 – 100.0** | `HIGH_EXECUTION_RISK` | `False` | *"Issue formal show-cause notice to [Constituency] District Authority and order mandatory on-site physical verification by District Vigilance Officer."* |
| **30.0 – 69.9** | `MODERATE_RISK` | `False` | *"Request expedited milestone status update from Implementing District Authority."* |
| **0.0 – 29.9** | `COMPLIANT_LOW_RISK` | `True` | *"Standard audit sign-off and Final Completion Certificate (FCC) approved."* |

---

## 5. Sample Input & Output JSON Schema

### Live API Input (`POST /api/predict/work-delay`)
```json
{
  "work_id": "WS/MP620/2024-2025/133166",
  "work_description": "Construction of Community Bhavan at Navalgund TQ Belavatagi Village",
  "work_category": "Normal/Others",
  "state": "Karnataka",
  "mp_name": "Pralhad Venkatesh Joshi",
  "constituency": "DHARWAD",
  "ida": "DHARWAD(DEPUTY COMMISSIONER DHARWAR_IDA)",
  "recommended_amount": 497185.0,
  "recommended_date": "2024-07-08",
  "sanction_amount": 497185.0,
  "sanction_date": "2024-07-09",
  "work_status": "Physical Inspection",
  "amount_disbursed": 448127.0,
  "completion_date": null,
  "has_photo_evidence": false
}
```

### Processed Output Payload Returned to Frontend
```json
{
  "work_id": "WS/MP620/2024-2025/133166",
  "execution_risk_score": 88.5,
  "risk_level": "HIGH_EXECUTION_RISK",
  "is_compliant": false,
  "lifecycle_metrics": {
    "recommended_amount": 497185.0,
    "sanction_amount": 497185.0,
    "amount_disbursed": 448127.0,
    "disbursement_pct": "90.13%",
    "current_stage": "Physical Inspection",
    "approval_latency_days": 1,
    "current_project_age_days": 416,
    "is_stalled": true,
    "has_photo_evidence": false
  },
  "flagged_reasons": [
    "Chronic Project Stalling: Project has been in 'Physical Inspection' stage for 416 days (national threshold: 180 days).",
    "Missing Mandatory Photographic Proof: 90.13% of funds disbursed with NO geo-tagged inspection photos uploaded.",
    "Blocked Public Capital: ₹4.48 Lakh disbursed without final project closure certificate."
  ],
  "explainability_tags": [
    "CHRONIC_STALLING_365D",
    "ZERO_PHOTO_EVIDENCE",
    "PHYSICAL_INSPECTION_BOTTLENECK"
  ],
  "recommended_action": "Issue formal show-cause notice to DHARWAD District Authority and order mandatory on-site physical verification by District Vigilance Officer."
}
```

---

## 6. Frontend UI Component Mapping

| Model Output Field | Target Frontend UI Element | Presentation in Dashboard |
| :--- | :--- | :--- |
| `execution_risk_score` | **Risk Rating Radial Meter** | Crimson gauge meter (`88.5 / 100`) |
| `risk_level` | **Severity Pill** | Red Badge (`HIGH EXECUTION RISK`) / Amber Badge / Green Badge |
| `current_stage` & `current_project_age_days` | **Milestone Timeline Strip** | Stage text with active days (`Physical Inspection — 416 Days Active`) |
| `is_stalled` & `has_photo_evidence` | **Compliance Pills** | `Stalled > 180 Days` warning badge & `Zero Photo Proof` alert tag |
| `flagged_reasons` | **Model Audit Infractions** | Bullet list of exact governance violations |
| `explainability_tags` | **Explainability Chips** | Tags (`#CHRONIC_STALLING_365D`, `#ZERO_PHOTO_EVIDENCE`) |
| `recommended_action` | **AI Governance Box** | Light teal callout box showing administrative advice |

---

## 7. Technical Summary

- **ML Algorithm:** Isolation Forest (`n_estimators=300`, `contamination=0.05`) + `StandardScaler`.
- **Hybrid Scorer:** Blends Isolation Forest decision function with statutory MPLADS governance rules.
- **Features (9D):** Approval latency, active project age, total lead time, chronic stalling indicator, photo penalty, status risk factor, cost escalation ratio, MP completion ratio, latency deviation.
- **Model Training Script:** `train_execution_model.py` → exports `models/execution_delay_model.joblib`.
- **Inference Speed:** < 1ms per project; vectorized batch scoring across thousands of works.
