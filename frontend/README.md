# PitchIQ AI — Frontend

Next.js 14 (App Router) + TypeScript + Tailwind + Recharts dashboard for the PitchIQ FastAPI backend.

## Run

```bash
npm install
npm run dev        # http://localhost:3000 (development)
# or
npm run build && npm run start
```

The backend must be running (default `http://localhost:8000`, see `../backend`). All `/api/*` requests are
proxied by Next.js rewrites to `PITCHIQ_API_URL`:

```bash
PITCHIQ_API_URL=http://localhost:8000 npm run dev
```

Background jobs started from the UI (video discovery, availability checks, benchmarks) need a worker:
`python -m pitchiq.jobs.worker` in `backend/`.

## Pages

Dashboard (blind prediction replay) · Pitcher Analytics (observed frequencies) · Game Simulator · Video Library ·
Prediction Challenge · Model Performance · Model Lab · Engineering Console · Settings.

Every number on screen comes from an API response; before a pitch is revealed the UI never receives its label
or video link. The Standard/Engineering toggle (top right) is stored in localStorage.
