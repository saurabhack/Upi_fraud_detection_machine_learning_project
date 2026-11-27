import { useEffect, useState } from 'react'
import {
  Activity,
  AlertTriangle,
  BarChart3,
  CheckCircle2,
  Download,
  FileSpreadsheet,
  ShieldAlert,
  ShieldCheck,
  Sparkles,
  Upload,
} from 'lucide-react'

const API_BASE = import.meta.env.VITE_API_URL || ''

const DEFAULT_FORM = {
  transaction_type: 'P2P',
  merchant_category: 'Food',
  amount_inr: 450,
  transaction_status: 'SUCCESS',
  sender_age_group: '26-35',
  receiver_age_group: '26-35',
  sender_state: 'Maharashtra',
  sender_bank: 'SBI',
  receiver_bank: 'HDFC',
  device_type: 'Android',
  network_type: '4G',
  hour: 19,
  day_of_week: 'Monday',
  model_type: 'xgboost',
}

const PRESETS = {
  normal: {
    ...DEFAULT_FORM,
    transaction_type: 'P2M',
    merchant_category: 'Food',
    amount_inr: 320,
    transaction_status: 'SUCCESS',
    device_type: 'Android',
    network_type: '5G',
    hour: 19,
    day_of_week: 'Wednesday',
  },
  highValue: {
    ...DEFAULT_FORM,
    transaction_type: 'P2M',
    merchant_category: 'Education',
    amount_inr: 15500,
    transaction_status: 'SUCCESS',
    device_type: 'iOS',
    network_type: 'WiFi',
    hour: 14,
    day_of_week: 'Tuesday',
  },
  suspicious: {
    ...DEFAULT_FORM,
    transaction_type: 'P2P',
    merchant_category: 'Shopping',
    amount_inr: 28000,
    transaction_status: 'FAILED',
    sender_bank: 'SBI',
    receiver_bank: 'Yes Bank',
    device_type: 'Web',
    network_type: '3G',
    hour: 2,
    day_of_week: 'Sunday',
  },
}

const SAMPLE_CSV = `transaction_type,merchant_category,amount_inr,transaction_status,sender_age_group,receiver_age_group,sender_state,sender_bank,receiver_bank,device_type,network_type,hour,day_of_week
P2P,Food,320,SUCCESS,26-35,26-35,Maharashtra,SBI,SBI,Android,5G,19,Monday
P2M,Education,14500,SUCCESS,36-45,46-55,Karnataka,HDFC,ICICI,iOS,WiFi,14,Tuesday
P2P,Shopping,26000,FAILED,18-25,46-55,Delhi,SBI,Axis,Web,3G,2,Sunday
Bill Payment,Utilities,1850,SUCCESS,26-35,36-45,Gujarat,ICICI,HDFC,Android,4G,9,Friday`

export default function App() {
  const [activeTab, setActiveTab] = useState('predict')
  const [health, setHealth] = useState({ ok: false, checked: false })
  const [metadata, setMetadata] = useState(null)
  const [form, setForm] = useState(DEFAULT_FORM)
  const [prediction, setPrediction] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const [batchFile, setBatchFile] = useState(null)
  const [batchModel, setBatchModel] = useState('xgboost')
  const [batchResult, setBatchResult] = useState(null)
  const [batchLoading, setBatchLoading] = useState(false)

  useEffect(() => {
    async function loadInitial() {
      try {
        const [hRes, mRes] = await Promise.all([
          fetch(`${API_BASE}/api/health`),
          fetch(`${API_BASE}/api/metadata`),
        ])
        if (hRes.ok) {
          const hData = await hRes.json()
          setHealth({ ok: Boolean(hData.models_loaded), checked: true })
        } else {
          setHealth({ ok: false, checked: true })
        }
        if (mRes.ok) {
          setMetadata(await mRes.json())
        }
      } catch {
        setHealth({ ok: false, checked: true })
      }
    }
    loadInitial()
  }, [])

  const handleFieldChange = (key, value) => {
    setForm((prev) => ({ ...prev, [key]: value }))
  }

  const handlePredict = async (e) => {
    e.preventDefault()
    setLoading(true)
    setError('')
    try {
      const res = await fetch(`${API_BASE}/api/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ...form,
          amount_inr: Number(form.amount_inr),
          hour: Number(form.hour),
        }),
      })
      if (!res.ok) {
        const err = await res.json().catch(() => ({}))
        throw new Error(err.detail || `Prediction failed (${res.status})`)
      }
      setPrediction(await res.json())
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const handleBatchSubmit = async () => {
    if (!batchFile) return
    setBatchLoading(true)
    setError('')
    try {
      const fd = new FormData()
      fd.append('file', batchFile)
      const res = await fetch(`${API_BASE}/api/predict/batch?model_type=${batchModel}`, {
        method: 'POST',
        body: fd,
      })
      if (!res.ok) {
        const err = await res.json().catch(() => ({}))
        throw new Error(err.detail || `Batch prediction failed (${res.status})`)
      }
      setBatchResult(await res.json())
    } catch (err) {
      setError(err.message)
    } finally {
      setBatchLoading(false)
    }
  }

  const downloadTextFile = (content, filename) => {
    const blob = new Blob([content], { type: 'text/csv;charset=utf-8;' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = filename
    a.click()
    URL.revokeObjectURL(url)
  }

  const exportBatchCsv = () => {
    if (!batchResult?.results?.length) return
    const headers = Object.keys(batchResult.results[0])
    const rows = batchResult.results.map((r) => headers.map((h) => r[h]).join(','))
    downloadTextFile([headers.join(','), ...rows].join('\n'), 'upi_fraud_predictions.csv')
  }

  const options = metadata?.categorical_options || {}
  const riskColor =
    prediction?.risk_level === 'HIGH'
      ? '#ef4444'
      : prediction?.risk_level === 'MEDIUM'
        ? '#f59e0b'
        : '#22c55e'

  return (
    <div className="app-shell">
      <header className="header">
        <div className="brand">
          <div className="brand-icon">
            <ShieldCheck size={24} />
          </div>
          <div>
            <h1>UPI Shield — Fraud Prediction &amp; Segmentation</h1>
            <p>FastAPI + React ML Platform (XGBoost · Logistic Regression · KMeans)</p>
          </div>
        </div>
        <div className={`status-pill ${health.ok ? '' : 'offline'}`}>
          <Activity size={15} />
          {!health.checked
            ? 'Connecting to API...'
            : health.ok
              ? 'Models Loaded & Ready'
              : 'Backend Offline'}
        </div>
      </header>

      <nav className="tabs">
        <button
          className={`tab-btn ${activeTab === 'predict' ? 'active' : ''}`}
          onClick={() => setActiveTab('predict')}
        >
          <ShieldAlert size={16} /> Live Fraud Predictor
        </button>
        <button
          className={`tab-btn ${activeTab === 'batch' ? 'active' : ''}`}
          onClick={() => setActiveTab('batch')}
        >
          <FileSpreadsheet size={16} /> Batch CSV Scoring
        </button>
        <button
          className={`tab-btn ${activeTab === 'analytics' ? 'active' : ''}`}
          onClick={() => setActiveTab('analytics')}
        >
          <BarChart3 size={16} /> Model &amp; Cluster Analytics
        </button>
      </nav>

      {error && <div className="error-box">{error}</div>}

      {activeTab === 'predict' && (
        <div className="grid-2">
          <form className="card" onSubmit={handlePredict}>
            <div className="card-header">
              <div className="card-title">
                <Sparkles size={18} color="#38bdf8" /> Transaction Parameters
              </div>
              <div className="presets">
                <button
                  type="button"
                  className="preset-btn"
                  onClick={() => setForm(PRESETS.normal)}
                >
                  Normal P2M
                </button>
                <button
                  type="button"
                  className="preset-btn"
                  onClick={() => setForm(PRESETS.highValue)}
                >
                  High-Value
                </button>
                <button
                  type="button"
                  className="preset-btn"
                  onClick={() => setForm(PRESETS.suspicious)}
                >
                  Late-Night Risk
                </button>
              </div>
            </div>

            <div className="form-grid">
              <div className="field">
                <label>Amount (INR)</label>
                <input
                  type="number"
                  min="1"
                  step="any"
                  value={form.amount_inr}
                  onChange={(e) => handleFieldChange('amount_inr', e.target.value)}
                  required
                />
              </div>

              <div className="field">
                <label>Hour of Day (0–23)</label>
                <input
                  type="number"
                  min="0"
                  max="23"
                  value={form.hour}
                  onChange={(e) => handleFieldChange('hour', e.target.value)}
                  required
                />
              </div>

              <div className="field">
                <label>ML Model</label>
                <select
                  value={form.model_type}
                  onChange={(e) => handleFieldChange('model_type', e.target.value)}
                >
                  <option value="xgboost">XGBoost Classifier</option>
                  <option value="logistic_regression">Logistic Regression</option>
                </select>
              </div>

              {[
                ['transaction_type', 'Transaction Type'],
                ['merchant_category', 'Merchant Category'],
                ['transaction_status', 'Status'],
                ['sender_bank', 'Sender Bank'],
                ['receiver_bank', 'Receiver Bank'],
                ['device_type', 'Device Type'],
                ['network_type', 'Network Type'],
                ['sender_state', 'Sender State'],
                ['sender_age_group', 'Sender Age'],
                ['receiver_age_group', 'Receiver Age'],
                ['day_of_week', 'Day of Week'],
              ].map(([key, label]) => (
                <div className="field" key={key}>
                  <label>{label}</label>
                  <select
                    value={form[key]}
                    onChange={(e) => handleFieldChange(key, e.target.value)}
                  >
                    {(options[key]?.length ? options[key] : [form[key]]).map((opt) => (
                      <option key={opt} value={opt}>
                        {opt}
                      </option>
                    ))}
                  </select>
                </div>
              ))}
            </div>

            <button className="primary-btn" type="submit" disabled={loading}>
              {loading ? 'Scoring Transaction...' : 'Analyze Fraud Risk & Segment'}
            </button>
          </form>

          <div className="card">
            <div className="card-header">
              <div className="card-title">Prediction Result</div>
              {prediction && (
                <span className="feat-pill">
                  Model: <strong>{prediction.model_used}</strong>
                </span>
              )}
            </div>

            {!prediction ? (
              <p style={{ color: 'var(--muted)', fontSize: '0.92rem' }}>
                Submit a transaction on the left (or click a preset) to compute real-time fraud
                probability, KMeans customer segment, and behavioral risk signals.
              </p>
            ) : (
              <>
                <div className={`risk-banner ${prediction.risk_level}`}>
                  <div className="risk-top">
                    <div>
                      <div style={{ fontSize: '0.8rem', color: 'var(--muted)' }}>
                        Fraud Risk Score
                      </div>
                      <div className="risk-score">{prediction.risk_score}%</div>
                    </div>
                    <span className={`badge ${prediction.risk_level}`}>
                      {prediction.risk_level} RISK
                    </span>
                  </div>
                  <div className="progress-track">
                    <div
                      className="progress-fill"
                      style={{
                        width: `${Math.max(prediction.risk_score, 3)}%`,
                        background: riskColor,
                      }}
                    />
                  </div>
                  <div style={{ fontSize: '0.84rem', color: 'var(--muted)' }}>
                    Raw Probability: <strong>{(prediction.fraud_probability * 100).toFixed(2)}%</strong>{' '}
                    · Verdict:{' '}
                    <strong>{prediction.is_fraud ? 'Flagged for Review' : 'Likely Legitimate'}</strong>
                  </div>
                </div>

                <div className="segment-box">
                  <div className="label">KMeans Customer Segment (Cluster {prediction.cluster_id})</div>
                  <div className="value">{prediction.customer_segment}</div>
                </div>

                <div style={{ fontSize: '0.85rem', fontWeight: 600, marginBottom: '8px' }}>
                  Behavioral Risk Signals
                </div>
                <div className="factors-list">
                  {prediction.risk_factors.map((rf, idx) => (
                    <div className="factor-item" key={idx}>
                      <div className="factor-head">
                        <span>{rf.label}</span>
                        <span style={{ textTransform: 'uppercase', fontSize: '0.72rem', color: 'var(--accent)' }}>
                          {rf.impact}
                        </span>
                      </div>
                      <div className="factor-detail">{rf.detail}</div>
                    </div>
                  ))}
                </div>

                <div style={{ fontSize: '0.85rem', fontWeight: 600, marginBottom: '8px' }}>
                  Engineered Features Sent to Pipeline
                </div>
                <div className="pill-grid">
                  {Object.entries(prediction.engineered_features).map(([k, v]) => (
                    <span className="feat-pill" key={k}>
                      {k}: <strong>{String(v)}</strong>
                    </span>
                  ))}
                </div>
              </>
            )}
          </div>
        </div>
      )}

      {activeTab === 'batch' && (
        <div className="card">
          <div className="card-header">
            <div className="card-title">
              <Upload size={18} color="#38bdf8" /> Batch CSV Fraud Scoring
            </div>
            <div style={{ display: 'flex', gap: '10px', flexWrap: 'wrap' }}>
              <button
                type="button"
                className="secondary-btn"
                onClick={() => downloadTextFile(SAMPLE_CSV, 'sample_upi_transactions.csv')}
              >
                <Download size={15} /> Sample CSV
              </button>
              {batchResult && (
                <button type="button" className="secondary-btn" onClick={exportBatchCsv}>
                  <Download size={15} /> Export Results
                </button>
              )}
            </div>
          </div>

          <div className="upload-zone">
            <input
              type="file"
              accept=".csv"
              onChange={(e) => setBatchFile(e.target.files?.[0] || null)}
              style={{ marginBottom: '12px' }}
            />
            <div style={{ fontSize: '0.85rem', color: 'var(--muted)' }}>
              Upload a CSV with transaction columns (up to 5,000 rows per batch).
            </div>
          </div>

          <div style={{ display: 'flex', gap: '12px', alignItems: 'center', marginBottom: '20px' }}>
            <select
              value={batchModel}
              onChange={(e) => setBatchModel(e.target.value)}
              style={{
                background: 'var(--bg)',
                color: 'var(--text)',
                border: '1px solid var(--border)',
                borderRadius: '10px',
                padding: '10px 12px',
              }}
            >
              <option value="xgboost">XGBoost Classifier</option>
              <option value="logistic_regression">Logistic Regression</option>
            </select>
            <button
              type="button"
              className="primary-btn"
              style={{ marginTop: 0, width: 'auto', padding: '10px 20px' }}
              disabled={!batchFile || batchLoading}
              onClick={handleBatchSubmit}
            >
              {batchLoading ? 'Scoring CSV...' : 'Run Batch Scoring'}
            </button>
          </div>

          {batchResult && (
            <>
              <div className="kpi-grid">
                <div className="kpi-card">
                  <div className="kpi-label">Transactions Scored</div>
                  <div className="kpi-val">{batchResult.total_count}</div>
                </div>
                <div className="kpi-card">
                  <div className="kpi-label">Flagged Fraud</div>
                  <div className="kpi-val" style={{ color: '#f87171' }}>
                    {batchResult.flagged_fraud_count}
                  </div>
                </div>
                <div className="kpi-card">
                  <div className="kpi-label">High Risk</div>
                  <div className="kpi-val" style={{ color: '#fbbf24' }}>
                    {batchResult.high_risk_count}
                  </div>
                </div>
              </div>

              <div className="table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th>#</th>
                      <th>Amount (₹)</th>
                      <th>Type</th>
                      <th>Category</th>
                      <th>Bank</th>
                      <th>Hour</th>
                      <th>Risk Score</th>
                      <th>Tier</th>
                      <th>Customer Segment</th>
                    </tr>
                  </thead>
                  <tbody>
                    {batchResult.results.slice(0, 100).map((r) => (
                      <tr key={r.row_index}>
                        <td>{r.row_index}</td>
                        <td>₹{r.amount_inr.toLocaleString()}</td>
                        <td>{r.transaction_type}</td>
                        <td>{r.merchant_category}</td>
                        <td>{r.sender_bank}</td>
                        <td>{r.hour}:00</td>
                        <td>{r.risk_score}%</td>
                        <td>
                          <span className={`badge ${r.risk_level}`}>{r.risk_level}</span>
                        </td>
                        <td>{r.customer_segment}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </>
          )}
        </div>
      )}

      {activeTab === 'analytics' && metadata && (
        <div>
          <div className="kpi-grid">
            <div className="kpi-card">
              <div className="kpi-label">Dataset Transactions</div>
              <div className="kpi-val">
                {metadata.dataset_stats.total_transactions.toLocaleString()}
              </div>
            </div>
            <div className="kpi-card">
              <div className="kpi-label">Confirmed Fraud Cases</div>
              <div className="kpi-val" style={{ color: '#f87171' }}>
                {metadata.dataset_stats.fraud_count.toLocaleString()} (
                {metadata.dataset_stats.fraud_rate_pct}%)
              </div>
            </div>
            <div className="kpi-card">
              <div className="kpi-label">Average Amount</div>
              <div className="kpi-val">
                ₹{metadata.dataset_stats.avg_amount_inr.toLocaleString()}
              </div>
            </div>
            <div className="kpi-card">
              <div className="kpi-label">Median Amount</div>
              <div className="kpi-val">
                ₹{metadata.dataset_stats.median_amount_inr.toLocaleString()}
              </div>
            </div>
          </div>

          <div className="card" style={{ marginBottom: '20px' }}>
            <div className="card-header">
              <div className="card-title">
                <CheckCircle2 size={18} color="#22c55e" /> Trained Fraud Models (80/20 Stratified Split)
              </div>
            </div>
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>Model</th>
                    <th>PR-AUC</th>
                    <th>ROC-AUC</th>
                    <th>Accuracy</th>
                    <th>Fraud Recall</th>
                    <th>Fraud Precision</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.values(metadata.metrics).map((m) => (
                    <tr key={m.name}>
                      <td>
                        <strong>{m.name}</strong>
                      </td>
                      <td>{m.pr_auc}</td>
                      <td>{m.roc_auc}</td>
                      <td>{(m.accuracy * 100).toFixed(2)}%</td>
                      <td>{(m.fraud_recall * 100).toFixed(2)}%</td>
                      <td>{(m.fraud_precision * 100).toFixed(2)}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          <div className="card">
            <div className="card-header">
              <div className="card-title">
                <AlertTriangle size={18} color="#38bdf8" /> KMeans Customer Segments (K = 5)
              </div>
            </div>
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>Cluster</th>
                    <th>Segment Name</th>
                    <th>Transactions</th>
                    <th>Share</th>
                    <th>Median (₹)</th>
                    <th>Avg Hour</th>
                    <th>Top Category</th>
                    <th>Fraud Rate</th>
                  </tr>
                </thead>
                <tbody>
                  {metadata.cluster_profiles.map((c) => (
                    <tr key={c.cluster_id}>
                      <td>#{c.cluster_id}</td>
                      <td>
                        <strong>{c.segment_name}</strong>
                      </td>
                      <td>{c.txn_count.toLocaleString()}</td>
                      <td>{c.share_pct}%</td>
                      <td>₹{c.median_amount_inr.toLocaleString()}</td>
                      <td>{c.avg_hour}:00</td>
                      <td>{c.top_category}</td>
                      <td>{c.fraud_rate_pct}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
