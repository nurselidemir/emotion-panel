import { useEffect, useMemo, useState } from "react";
import "./index.css";

const API_URL = process.env.REACT_APP_API_URL || "http://localhost:8000";


const ACCEPT = { image: ".png", video: ".mp4", audio: ".wav" };

function Section({ title, type, models, model, setModel, files, setFiles, perf }) {
  const onFiles = (e) => setFiles(Array.from(e.target.files || []));

  const options = (models || []).map((m) =>
    typeof m === "string" ? { name: m, accuracy: perf[m] } : { name: m?.name, accuracy: perf[m?.name] }
  );

  return (
    <div className="card">
      <div className="card-head">
        <h2>{title}</h2>
        <span className="muted">({ACCEPT[type]} dosyalar)</span>
      </div>

      <div className="form-row">
        <label>Model</label>
        <select value={model} onChange={(e) => setModel(e.target.value)}>
          <option value="">— model seç —</option>
          {options.map((o) => (
            <option key={o.name} value={o.name}>
              {o.name} {o.accuracy ? `(${o.accuracy}%)` : ""}
            </option>
          ))}
        </select>
        {model && (perf[model] !== undefined) && (
          <div className="hint">Accuracy: {perf[model]}%</div>
        )}
      </div>

      <div className="form-row">
        <label>Dosyalar</label>
        <input type="file" multiple accept={ACCEPT[type]} onChange={onFiles} />
        <div className="hint">
          Seçilen dosya: {files.length ? files.map((f) => f.name).join(", ") : "—"}
        </div>
      </div>
    </div>
  );
}

export default function App() {
  const [modelsByCat, setModelsByCat] = useState({ image: [], video: [], audio: [] });
  const [perfMap, setPerfMap] = useState({}); // { modelName: accuracyNumber }

  const [imageModel, setImageModel] = useState("");
  const [videoModel, setVideoModel] = useState("");
  const [audioModel, setAudioModel] = useState("");

  const [imageFiles, setImageFiles] = useState([]);
  const [videoFiles, setVideoFiles] = useState([]);
  const [audioFiles, setAudioFiles] = useState([]);

  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState([]);
  const [error, setError] = useState("");

  
  useEffect(() => {
    Promise.all([
      fetch(`${API_URL}/models`).then((r) => r.json()),
      fetch(`${API_URL}/models/performance`).then((r) => r.json()),
    ])
      .then(([modelsData, perfData]) => {
        setModelsByCat(modelsData);
        
        const acc = {};
        Object.entries(perfData || {}).forEach(([name, val]) => {
          if (val && typeof val.accuracy !== "undefined") acc[name] = Number(val.accuracy);
        });
        setPerfMap(acc);
      })
      .catch(() => setError("Modeller alınamadı. Backend çalışıyor mu?"));
  }, []);

  const jobs = useMemo(
    () => [
      { cat: "image", model: imageModel, files: imageFiles },
      { cat: "video", model: videoModel, files: videoFiles },
      { cat: "audio", model: audioModel, files: audioFiles },
    ],
    [imageModel, videoModel, audioModel, imageFiles, videoFiles, audioFiles]
  );

  const analyze = async () => {
    setError("");

    
    const queue = [];
    for (const j of jobs) {
      if (!j.files.length) continue;
      if (!j.model) {
        setError(`${j.cat} için model seç.`);
        return;
      }
      for (const f of j.files) {
        const form = new FormData();
        form.append("file", f);
        form.append("model_name", j.model);
        form.append("category", j.cat);
        queue.push({ form, meta: { filename: f.name, category: j.cat, model: j.model } });
      }
    }
    if (!queue.length) {
      setError("Lütfen en az bir dosya seçin.");
      return;
    }

    
    setResults([]);
    setLoading(true);

    for (const q of queue) {
      try {
        const res = await fetch(`${API_URL}/predict`, { method: "POST", body: q.form });
        if (!res.ok) {
          const errBody = await res.json().catch(() => ({}));
          throw new Error(errBody.detail || "İstek başarısız");
        }
        const data = await res.json();
        setResults((prev) => [
          ...prev,
          {
            filename: q.meta.filename,
            category: q.meta.category,
            model: data.model_name,
            predicted_emotion: data.predicted_emotion,
          },
        ]);
      } catch (e) {
        setResults((prev) => [
          ...prev,
          {
            filename: q.meta.filename,
            category: q.meta.category,
            model: q.meta.model,
            predicted_emotion: `Hata: ${e.message}`,
          },
        ]);
      }
    }

    setLoading(false);
  };

  return (
    <div className="page">
      <header className="header">
        <h1>Emotion Panel</h1>
      </header>

      <div className="grid3">
        <Section
          title="Görsel"
          type="image"
          models={modelsByCat.image}
          model={imageModel}
          setModel={setImageModel}
          files={imageFiles}
          setFiles={setImageFiles}
          perf={perfMap}
        />
        <Section
          title="Video"
          type="video"
          models={modelsByCat.video}
          model={videoModel}
          setModel={setVideoModel}
          files={videoFiles}
          setFiles={setVideoFiles}
          perf={perfMap}
        />
        <Section
          title="Ses"
          type="audio"
          models={modelsByCat.audio}
          model={audioModel}
          setModel={setAudioModel}
          files={audioFiles}
          setFiles={setAudioFiles}
          perf={perfMap}
        />
      </div>

      <div className="actions">
        <button className="btn" onClick={analyze} disabled={loading}>
          {loading ? "Analiz yapılıyor..." : "Analizi Başlat"}
        </button>
      </div>

      {error && <div className="alert">{error}</div>}

      {results.length > 0 && (
        <section className="card">
          <h2>Sonuçlar</h2>
          <div className="table">
            <div className="tr th">
              <div>Dosya</div>
              <div>Tür</div>
              <div>Model</div>
              <div>Tahmin</div>
            </div>
            {results.map((r, i) => (
              <div className="tr" key={i}>
                <div className="td">{r.filename}</div>
                <div className="td">{r.category}</div>
                <div className="td">
                  {r.model}
                  {perfMap[r.model] !== undefined ? ` (${perfMap[r.model]}%)` : ""}
                </div>
                <div className="td">{r.predicted_emotion}</div>
              </div>
            ))}
          </div>
        </section>
      )}
    </div>
  );
}
