# AI Backend (Flask + ML)

## 1) Install
```bash
cd ai_backend
pip install -r requirements.txt
```

## 2) Train model (recommended)
```bash
python train_model.py
```

This creates:
- `model/student_score_model.joblib`
- `model/model_info.json`

## 3) Run Flask API
```bash
python app.py
```

API will be available at:
- `GET /api/health`
- `GET /api/model-info`
- `POST /api/predict`

## 4) Connect with frontend

Frontend (`app.js`) already calls:
- `/api/model-info`
- `/api/predict`

If backend is hosted separately (example: Render), set this in `index.html` before `app.js`:

```html
<script>
  window.__AI_API_BASE = "https://YOUR-RENDER-SERVICE.onrender.com";
</script>
```

## 5) Deploy notes

- Static site: GitHub Pages (your current setup)
- Flask API: Render / Replit / Railway
- Add CORS (already enabled in `app.py`)
