
import React, { useState } from 'react'
import { createRoot } from 'react-dom/client'
import {
  Upload,
  ScanEye,
  ShieldCheck,
  Activity,
  Video,
  LoaderCircle,
  AlertCircle,
  Play,
  Download,
  Users,
  Gauge,
  Film,
  Eye
} from 'lucide-react'

import './style.css'

const API = import.meta.env.VITE_API_URL || 'http://localhost:8000'

function App() {
  const [file, setFile] = useState(null)
  const [confidence, setConfidence] = useState(0.35)
  const [maxAge, setMaxAge] = useState(90)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [result, setResult] = useState(null)

  async function startTracking(e) {
    e.preventDefault()

    if (!file) {
      setError('Please select a video before starting tracking.')
      return
    }

    setLoading(true)
    setError('')
    setResult(null)

    const body = new FormData()
    body.append('video', file)

    try {
      const response = await fetch(
        `${API}/api/track?confidence=${confidence}&max_age=${maxAge}`,
        {
          method: 'POST',
          body
        }
      )

      const data = await response.json()

      if (!response.ok) {
        throw new Error(data.detail || 'Tracking request failed.')
      }

      setResult(data)
    } catch (err) {
      setError(
        err.message || 'Could not connect to the backend. Check whether FastAPI is running.'
      )
    } finally {
      setLoading(false)
    }
  }

  function handleFileChange(e) {
    const selectedFile = e.target.files?.[0] || null

    setFile(selectedFile)
    setError('')
    setResult(null)
  }

  const identities = result?.identity_summary || []

  const totalObservations = identities.reduce(
    (sum, item) => sum + (Number(item.observations) || 0),
    0
  )

  const videoUrl = result?.video_url
    ? `${API}${result.video_url}`
    : ''

  return (
    <div className="app">

      {/* HEADER */}
      <header className="topbar">
        <div className="brand">
          <div className="brand-icon">
            <ScanEye size={22} />
          </div>

          <span>
            TRACK<span className="accent">/</span>UNTRACKABLE
          </span>
        </div>

        <span className="pill">
          <span className="dot" />
          COMPUTER VISION LAB
        </span>
      </header>

      <main>

        {/* HERO */}
        <section className="hero">
          <div className="eyebrow">
            <span />
            LONG-HORIZON MULTI-OBJECT TRACKING
          </div>

          <h1>
            Track the <em>untrackable.</em>
          </h1>

          <p className="intro">
            Upload a video. Detect and track multiple objects.
            Follow their movement and analyze their identities
            across video frames.
          </p>
        </section>

        {/* WORKSPACE */}
        <section className="workspace">

          {/* INPUT PANEL */}
          <form
            className="panel upload-panel"
            onSubmit={startTracking}
          >
            <div className="panel-heading">
              <div>
                <span className="step">01 / INPUT</span>
                <h2>Video source</h2>
              </div>

              <Video size={21} />
            </div>

            <label className="dropzone">

              <input
                type="file"
                accept="video/mp4,video/avi,video/quicktime,video/x-matroska,.mkv"
                onChange={handleFileChange}
              />

              <div className="upload-icon">
                <Upload size={25} />
              </div>

              <strong>
                {file ? file.name : 'Drop your video here'}
              </strong>

              <span>
                {file
                  ? `${(file.size / 1024 / 1024).toFixed(1)} MB · Ready to process`
                  : 'Click to browse · MP4, MOV, AVI, MKV'}
              </span>

            </label>

            {/* SETTINGS */}
            <div className="settings">

              <div className="setting-label">
                <label htmlFor="confidence">
                  Detection confidence
                </label>

                <b>{Number(confidence).toFixed(2)}</b>
              </div>

              <input
                id="confidence"
                type="range"
                min="0.1"
                max="0.9"
                step="0.05"
                value={confidence}
                onChange={e => setConfidence(Number(e.target.value))}
              />

              <div className="setting-label">
                <label htmlFor="age">
                  Identity memory window
                </label>

                <b>{maxAge} frames</b>
              </div>

              <input
                id="age"
                type="range"
                min="15"
                max="300"
                step="15"
                value={maxAge}
                onChange={e => setMaxAge(Number(e.target.value))}
              />

              <p className="hint">
                How long a missing track remains available for
                possible identity recovery.
              </p>
            </div>

            {/* ERROR */}
            {error && (
              <div className="error" role="alert">
                <AlertCircle size={17} />
                {error}
              </div>
            )}

            {/* SUBMIT */}
            <button
              className="run-button"
              type="submit"
              disabled={loading || !file}
            >
              {loading ? (
                <>
                  <LoaderCircle className="spin" size={18} />
                  Processing video...
                </>
              ) : (
                <>
                  <Play size={17} fill="currentColor" />
                  Run identity tracking
                </>
              )}
            </button>

            <p className="privacy">
              Video is processed by your local backend instance.
            </p>

          </form>

          {/* RESULTS PANEL */}
          <section className="panel results-panel">

            <div className="panel-heading">
              <div>
                <span className="step">02 / ANALYSIS</span>
                <h2>Tracking output</h2>
              </div>

              <Activity size={21} />
            </div>

            {/* EMPTY STATE */}
            {!result && !loading && (
              <div className="empty">
                <div className="empty-art">
                  <ScanEye size={42} />
                </div>

                <strong>Awaiting video analysis</strong>

                <span>
                  Your annotated video and identity report
                  will appear here.
                </span>
              </div>
            )}

            {/* LOADING STATE */}
            {loading && (
              <div className="empty">
                <LoaderCircle
                  className="spin accent-text"
                  size={38}
                />

                <strong>Following identities...</strong>

                <span>
                  Detection and frame-by-frame association
                  are running.
                </span>
              </div>
            )}

            {/* RESULT STATE */}
            {result && (
              <div className="result-content">

                {/* STATISTICS */}
                <div className="stats">

                  <div className="stat-card">
                    <Users size={17} />
                    <span>IDENTITIES</span>
                    <b>{result.total_ids ?? identities.length}</b>
                  </div>

                  <div className="stat-card">
                    <Film size={17} />
                    <span>FRAMES</span>
                    <b>{result.frames ?? 0}</b>
                  </div>

                  <div className="stat-card">
                    <Gauge size={17} />
                    <span>FRAME RATE</span>
                    <b>{result.fps ?? 0}</b>
                  </div>

                  <div className="stat-card">
                    <Eye size={17} />
                    <span>OBSERVATIONS</span>
                    <b>{totalObservations}</b>
                  </div>

                </div>

                {/* VIDEO */}
                <div className="section-label">
                  <h3>Annotated video</h3>
                  <span>TRACKING PLAYBACK</span>
                </div>

                <div className="video-wrap">
                  <video
                    controls
                    preload="metadata"
                    src={videoUrl}
                  >
                    Your browser does not support video playback.
                  </video>
                </div>

                {/* IDENTITY REGISTER */}
                <div className="identity-title">
                  <h3>Identity register</h3>

                  <span>
                    {identities.length} tracked IDs
                  </span>
                </div>

                <div className="identity-list">

                  {identities.length === 0 ? (
                    <div className="no-identities">
                      No identities were detected.
                    </div>
                  ) : (
                    identities.map(item => (
                      <div
                        className="identity-row"
                        key={item.id}
                      >
                        <div className="id-chip">
                          #{item.id}
                        </div>

                        <div className="identity-name">
                          <b>
                            {item.class_name || 'Unknown object'}
                          </b>

                          <span>
                            Frames {item.first_seen ?? '-'} – {item.last_seen ?? '-'}
                          </span>
                        </div>

                        <div className="observations">
                          {item.observations ?? 0}
                          <small>observations</small>
                        </div>
                      </div>
                    ))
                  )}

                </div>

                {/* DOWNLOAD */}
                {videoUrl && (
                  <a
                    className="download"
                    href={videoUrl}
                    download
                  >
                    <Download size={16} />
                    Download annotated video
                  </a>
                )}

                <p className="result-note">
                  Tracker IDs represent algorithm-generated tracks,
                  not verified real-world identities.
                </p>

              </div>
            )}

          </section>

        </section>

        {/* FEATURES */}
        <section className="feature-row">

          <div>
            <ShieldCheck />
            <b>Identity memory</b>
            <span>Retains recently lost tracks</span>
          </div>

          <div>
            <Activity />
            <b>Motion + appearance</b>
            <span>Combines location and visual cues</span>
          </div>

          <div>
            <ScanEye />
            <b>Confidence-aware</b>
            <span>Threshold-based detection</span>
          </div>

        </section>

        <footer>
          HACKATHON PROTOTYPE
          <span>·</span>
          YOLO + OpenCV + FastAPI + React
        </footer>

      </main>
    </div>
  )
}

createRoot(document.getElementById('root')).render(<App />)

