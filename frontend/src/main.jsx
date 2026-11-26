
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
  const [files, setFiles] = useState([])
  const [cameraIds, setCameraIds] = useState([])
  const [startTimes, setStartTimes] = useState([])
  const [multiCamera, setMultiCamera] = useState(false)
  const [confidence, setConfidence] = useState(0.1)
  const [maxAge, setMaxAge] = useState(90)
  const [countLinePosition, setCountLinePosition] = useState(50)
  const [inDirection, setInDirection] = useState('top_to_bottom')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [result, setResult] = useState(null)

  async function startTracking(e) {
    e.preventDefault()

    if (!files.length || (multiCamera && files.length < 2)) {
      setError(
        multiCamera
          ? 'Select at least two camera videos for cross-camera tracking.'
          : 'Please select a video before starting tracking.'
      )
      return
    }

    setLoading(true)
    setError('')
    setResult(null)

    const body = new FormData()
    let endpoint = '/api/track'
    if (multiCamera) {
      endpoint = '/api/track/multi-camera'
      files.forEach(file => body.append('videos', file))
      cameraIds.forEach(cameraId => body.append('camera_ids', cameraId))
      startTimes.forEach(startTime => body.append('start_times', startTime || '0'))
      body.append('count_line_position', String(countLinePosition / 100))
      body.append('in_direction', inDirection)
    } else {
      body.append('video', files[0])
    }

    try {
      const query = new URLSearchParams({
        confidence: String(confidence),
        max_age: String(maxAge),
        count_line_position: String(countLinePosition / 100),
        in_direction: inDirection
      })
      const response = await fetch(
        `${API}${endpoint}?${query}`,
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
    const selectedFiles = Array.from(e.target.files || [])
    setFiles(multiCamera ? selectedFiles : selectedFiles.slice(0, 1))
    setCameraIds(selectedFiles.map((_, index) => `camera_${index + 1}`))
    setStartTimes(selectedFiles.map(() => '0'))
    setError('')
    setResult(null)
  }

  function changeMode(enabled) {
    setMultiCamera(enabled)
    setFiles([])
    setCameraIds([])
    setStartTimes([])
    setError('')
    setResult(null)
  }

  function updateCameraId(index, value) {
    setCameraIds(current =>
      current.map((cameraId, cameraIndex) =>
        cameraIndex === index ? value : cameraId
      )
    )
  }

  function updateStartTime(index, value) {
    setStartTimes(current =>
      current.map((startTime, timeIndex) =>
        timeIndex === index ? value : startTime
      )
    )
  }

  const cameraResults = result?.camera_results || (result ? [result] : [])
  const identities = cameraResults.flatMap(camera =>
    (camera.identity_summary || []).map(item => ({
      ...item,
      camera_id: item.camera_id || camera.camera_id
    }))
  )

  const totalObservations = identities.reduce(
    (sum, item) => sum + (Number(item.observations) || 0),
    0
  )

  const objectCounts = result?.object_counts || { total_unique: 0, by_class: {} }
  const totalFrames = cameraResults.reduce(
    (sum, camera) => sum + (Number(camera.frames) || 0),
    0
  )
  const totalUnique = result?.total_global_ids
    ?? result?.total_ids
    ?? objectCounts.total_unique
    ?? identities.length

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
            Upload a video. Detect and track people and common objects,
            including smaller objects farther from the camera.
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
                <h2>{multiCamera ? 'Camera sources' : 'Video source'}</h2>
              </div>

              <Video size={21} />
            </div>

            <div className="mode-switch" role="group" aria-label="Tracking mode">
              <label>
                <input
                  type="radio"
                  name="tracking-mode"
                  checked={!multiCamera}
                  onChange={() => changeMode(false)}
                />
                Single camera
              </label>
              <label>
                <input
                  type="radio"
                  name="tracking-mode"
                  checked={multiCamera}
                  onChange={() => changeMode(true)}
                />
                Multi-camera batch
              </label>
            </div>

            <label className="dropzone">

              <input
                type="file"
                accept="video/mp4,video/avi,video/quicktime,video/x-matroska,.mkv"
                multiple={multiCamera}
                onChange={handleFileChange}
              />

              <div className="upload-icon">
                <Upload size={25} />
              </div>

              <strong>
                {files.length
                  ? `${files.length} video${files.length === 1 ? '' : 's'} selected`
                  : 'Drop your video here'}
              </strong>

              <span>
                {files.length
                  ? files.map(file => file.name).join(' · ')
                  : 'Click to browse · MP4, MOV, AVI, MKV'}
              </span>

            </label>

            {multiCamera && files.length > 0 && (
              <div className="camera-list">
                {files.map((file, index) => (
                  <div className="camera-file-row" key={`${file.name}-${index}`}>
                    <span>{file.name}</span>
                    <input
                      type="text"
                      value={cameraIds[index] || ''}
                      maxLength={64}
                      aria-label={`Camera ID for ${file.name}`}
                      onChange={e => updateCameraId(index, e.target.value)}
                    />
                    <input
                      type="number"
                      min="0"
                      step="0.1"
                      value={startTimes[index] ?? '0'}
                      aria-label={`Start time offset for ${file.name} in seconds`}
                      onChange={e => updateStartTime(index, e.target.value)}
                    />
                  </div>
                ))}
                <p className="hint">
                  Use a different camera ID for each clip. Cross-camera matches
                  are based on visual appearance and should be reviewed. Set
                  offsets if the clips start at different times.
                </p>
              </div>
            )}

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
                min="0.05"
                max="0.9"
                step="0.05"
                value={confidence}
                onChange={e => setConfidence(Number(e.target.value))}
              />

              <p className="hint">
                Lower values can catch faint or distant objects, but may
                also add false detections.
              </p>

              <div className="setting-label">
                <label htmlFor="age">Track memory window</label>

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

              <div className="setting-label">
                <label htmlFor="count-line-position">Counting line position</label>
                <b>{countLinePosition}%</b>
              </div>
              <input
                id="count-line-position"
                type="range"
                min="10"
                max="90"
                step="5"
                value={countLinePosition}
                onChange={e => setCountLinePosition(Number(e.target.value))}
              />
              <p className="hint">
                IN direction is configurable below; OUT is the reverse crossing.
              </p>

              <div className="setting-label">
                <label htmlFor="in-direction">IN direction</label>
              </div>
              <select
                id="in-direction"
                value={inDirection}
                onChange={e => setInDirection(e.target.value)}
              >
                <option value="top_to_bottom">Top to bottom</option>
                <option value="bottom_to_top">Bottom to top</option>
              </select>
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
              disabled={loading || !files.length || (multiCamera && files.length < 2)}
            >
              {loading ? (
                <>
                  <LoaderCircle className="spin" size={18} />
                  Processing video...
                </>
              ) : (
                <>
                  <Play size={17} fill="currentColor" />
                  {multiCamera ? 'Track across cameras' : 'Run object tracking'}
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
                    <span>UNIQUE OBJECTS</span>
                    <b>{totalUnique}</b>
                  </div>

                  <div className="stat-card">
                    <Film size={17} />
                    <span>FRAMES</span>
                    <b>{totalFrames}</b>
                  </div>

                  <div className="stat-card">
                    <Gauge size={17} />
                    <span>{result.camera_count ? 'CAMERAS' : 'FRAME RATE'}</span>
                    <b>{result.camera_count ?? (result.fps ?? 0)}</b>
                  </div>

                  <div className="stat-card">
                    <Eye size={17} />
                    <span>DETECTIONS</span>
                    <b>{result.total_detections ?? totalObservations}</b>
                  </div>

                </div>

                <div className="count-summary">
                  <b>Unique objects by class</b>
                  <div>
                    {Object.entries(objectCounts.by_class || {}).map(([name, count]) => (
                      <span key={name}>{name}: {count}</span>
                    ))}
                    {!Object.keys(objectCounts.by_class || {}).length && (
                      <span>No objects counted</span>
                    )}
                  </div>
                </div>

                {cameraResults.map(camera => {
                  const videoUrl = camera.video_url ? `${API}${camera.video_url}` : ''
                  return (
                    <section className="camera-result" key={camera.camera_id}>
                      <div className="section-label">
                        <h3>Camera {camera.camera_id || 'video'}</h3>
                        <span>{camera.frames ?? 0} FRAMES</span>
                      </div>
                      <div className="video-wrap">
                        <video controls preload="metadata" src={videoUrl}>
                          Your browser does not support video playback.
                        </video>
                      </div>
                      <div className="flow-counts">
                        <b>Line-crossing counts</b>
                        <span>
                          IN {camera.flow_counts?.total_in ?? 0}
                          {' · '}
                          OUT {camera.flow_counts?.total_out ?? 0}
                        </span>
                        {Object.entries(camera.flow_counts?.by_class || {}).map(
                          ([name, counts]) => (
                            <small key={name}>
                              {name}: IN {counts.in}, OUT {counts.out}
                            </small>
                          )
                        )}
                      </div>
                      <div className="movement-alerts" role="status" aria-live="polite">
                        <b>Unusual movement alerts</b>
                        <small>
                          Person-only motion cues. Restricted-zone alerts are off.
                        </small>
                        {camera.movement_alerts?.length ? (
                          camera.movement_alerts.map(alert => (
                            <div
                              className="movement-alert"
                              key={`${alert.track_id}-${alert.alert_type}-${alert.frame}`}
                            >
                              <span>{alert.message}</span>
                              <small>
                                {alert.class_name} G{alert.global_id}
                                {' · '}
                                {alert.time_seconds}s
                                {' · '}
                                {alert.value} {alert.unit}
                              </small>
                            </div>
                          ))
                        ) : (
                          <small>No unusual movement detected.</small>
                        )}
                      </div>
                      {videoUrl && (
                        <a className="download" href={videoUrl} download>
                          <Download size={16} />
                          Download {camera.camera_id || 'annotated'} video
                        </a>
                      )}
                    </section>
                  )
                })}

                {result.cross_camera_matches?.length > 0 && (
                  <div className="cross-camera-summary">
                    <b>Cross-camera links</b>
                    {result.cross_camera_matches.map((match, index) => (
                      <span key={`${match.global_id}-${index}`}>
                        G{match.global_id}: {match.from_camera} → {match.to_camera}
                        {' '}({Math.round(match.appearance_similarity * 100)}% appearance match)
                      </span>
                    ))}
                  </div>
                )}

                {/* IDENTITY REGISTER */}
                <div className="identity-title">
                  <h3>Identity register</h3>

                  <span>
                    {identities.length} camera tracks
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
                        key={`${item.camera_id}-${item.id}`}
                      >
                        <div className="id-chip">
                          G{item.global_id ?? item.id}
                        </div>

                        <div className="identity-name">
                          <b>
                            {item.class_name || 'Unknown object'}
                          </b>

                          <span>
                            {item.camera_id} · Local #{item.id} · Frames {item.first_seen ?? '-'} – {item.last_seen ?? '-'}
                          </span>
                        </div>

                        <div className="observations">
                          {item.observations ?? 0}
                          <small>observations</small>
                        </div>
                        {item.movement_history?.length > 0 && (
                          <details className="movement-details">
                            <summary>Movement ({item.movement_history.length} samples)</summary>
                            <ol>
                              {item.movement_history.slice(-10).map(point => (
                                <li key={`${point.frame}-${point.time_seconds}`}>
                                  {point.time_seconds}s · ({point.center[0]}, {point.center[1]})
                                </li>
                              ))}
                            </ol>
                          </details>
                        )}
                      </div>
                    ))
                  )}

                </div>

                <p className="result-note">
                  Track IDs are algorithm-generated. Cross-camera appearance
                  links are estimates, not verified identities.
                </p>

              </div>
            )}

          </section>

        </section>

        {/* FEATURES */}
        <section className="feature-row">

          <div>
            <ShieldCheck />
            <b>Cross-camera links</b>
            <span>Matches appearance across uploaded camera clips</span>
          </div>

          <div>
            <Activity />
            <b>Movement history</b>
            <span>Stores sampled positions for each track</span>
          </div>

          <div>
            <ScanEye />
            <b>Automatic counting</b>
            <span>Counts unique tracks by object class</span>
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

