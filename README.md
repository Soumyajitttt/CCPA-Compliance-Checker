# CCPA Compliance Checker

**OPEN HACK 2026 — CSA, IISc**

A Dockerized FastAPI service that analyzes natural-language business practice descriptions and returns a structured JSON verdict indicating whether the practice violates the California Consumer Privacy Act (CCPA), along with the exact violated sections..

---

## Solution Overview

### Architecture

```
Natural Language Prompt
        │
        ▼
┌──────────────────────┐
│  Layer 1: Unrelated  │  ──► Short-circuit safe for non-privacy prompts
│  Prompt Detection    │
└──────────────────────┘
        │
        ▼
┌──────────────────────┐
│  Layer 2: Keyword    │  ──► Fast sweep across all 10 CCPA sections
│  Matching Engine     │       using 100+ hand-crafted violation keywords
└──────────────────────┘
        │
        ▼
┌──────────────────────┐
│  Layer 3: LLM        │  ──► google/flan-t5-base verifies & catches
│  Verification        │       nuanced violations keyword layer may miss
└──────────────────────┘
        │
        ▼
   Union of Results
        │
        ▼
  {"harmful": bool, "articles": [...]}
```

### Knowledge Base

The **complete verbatim text of all 65 pages** of the CCPA statute (Sections 1798.100 through 1798.199.100) is embedded directly in `main.py` as a Python string constant (`CCPA_FULL_TEXT`). This includes:

- Every section word-for-word
- All subsections, exemptions, and definitions
- The full definitions of "personal information", "sensitive personal information", "sell", "share", "consent", "business", "consumer", etc.
- Administrative enforcement, civil penalty, and anti-avoidance provisions

### Model

- **Model**: `google/flan-t5-base` (~250 MB, encoder-decoder T5 architecture)
- **Parameters**: ~250 million (well under the 8B limit)
- **HF Token**: Not required (fully public model)
- **Runtime**: CPU-only (GPU optional for faster inference)
- **Pre-downloaded**: Model weights are downloaded at Docker **build time**, not at inference time

### Detection Logic

1. **Keyword Layer**: 100+ keywords/phrases mapped to all 10 major CCPA sections (1798.100, 1798.105, 1798.106, 1798.110, 1798.115, 1798.120, 1798.121, 1798.125, 1798.130, 1798.135). Catches explicit violations instantly.
2. **LLM Layer**: A focused prompt with all 10 rules is sent to flan-t5-base, which outputs either `COMPLIANT` or `VIOLATION: Section 1798.XXX`. Catches nuanced or implicit violations.
3. **Union**: Both layers' results are merged. All unique violated sections are returned.

---

## Docker Run Command

```bash
docker run --gpus all -p 8000:8000 -e HF_TOKEN=your_token_here yourusername/ccpa-compliance:latest
```

**CPU-only (no GPU):**
```bash
docker run -p 8000:8000 yourusername/ccpa-compliance:latest
```

**With Docker Compose:**
```bash
docker compose up -d
```

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `HF_TOKEN` | No | Hugging Face access token. **Not required** for this project — `google/flan-t5-base` is a fully public model. Include only if swapping to a gated model. |

---

## GPU Requirements

| Requirement | Value |
|-------------|-------|
| Minimum GPU VRAM | None required |
| GPU recommended | Optional (speeds up inference) |
| CPU-only fallback | ✅ Fully supported |

The model runs comfortably on CPU. GPU will reduce inference time from ~2–5 seconds to ~0.2 seconds per request.

---

## Local Setup Instructions (Fallback — No Docker)

Use this only if Docker fails. **Manual deployment incurs a score penalty.**

### Step 1 — Install Python 3.11

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y python3.11 python3.11-venv python3-pip
```

Verify:
```bash
python3.11 --version
# Python 3.11.x
```

### Step 2 — Clone / extract the project

```bash
unzip ccpa_compliance.zip
cd ccpa_compliance
```

### Step 3 — Create a virtual environment

```bash
python3.11 -m venv venv
source venv/bin/activate
```

### Step 4 — Install dependencies

```bash
pip install -r requirements.txt
```

This installs (~500 MB):
- `fastapi`
- `uvicorn`
- `transformers`
- `torch`
- `sentencepiece`
- `pydantic`

### Step 5 — Pre-download the model (one-time, ~250 MB)

```bash
python3 -c "
from transformers import T5ForConditionalGeneration, T5Tokenizer
T5Tokenizer.from_pretrained('google/flan-t5-base')
T5ForConditionalGeneration.from_pretrained('google/flan-t5-base')
print('Model ready.')
"
```

### Step 6 — Start the server

```bash
cd app
python main.py
```

Expected output:
```
INFO:__main__:Loading model: google/flan-t5-base
INFO:__main__:Model loaded successfully on cpu
INFO:     Started server process [XXXX]
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

### Step 7 — Run the format validator (new terminal)

```bash
cd ..
python validate_format.py
```

---

## API Usage Examples

### Health Check

```bash
curl http://localhost:8000/health
```

Response:
```json
{"status": "ok"}
```

### Violation Detected — Selling Without Opt-Out

```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"prompt": "We sell customer browsing history to ad networks without notifying them."}'
```

Response:
```json
{"harmful": true, "articles": ["Section 1798.100", "Section 1798.120"]}
```

### Violation Detected — Ignoring Deletion Requests

```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"prompt": "When customers ask us to delete their data, we keep all records anyway."}'
```

Response:
```json
{"harmful": true, "articles": ["Section 1798.105"]}
```

### Violation Detected — Selling Minors' Data

```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"prompt": "We sell the personal data of our 14-year-old users without getting parental consent."}'
```

Response:
```json
{"harmful": true, "articles": ["Section 1798.120"]}
```

### Violation Detected — Discriminatory Pricing

```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"prompt": "We charge higher prices to customers who opted out of our data sharing program."}'
```

Response:
```json
{"harmful": true, "articles": ["Section 1798.125"]}
```

### Violation Detected — Missing Do Not Sell Link

```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Our website sells user data but has no Do Not Sell My Personal Information link on our homepage."}'
```

Response:
```json
{"harmful": true, "articles": ["Section 1798.135"]}
```

### No Violation — Compliant Business

```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"prompt": "We provide a clear privacy policy and honor all deletion requests within 45 days."}'
```

Response:
```json
{"harmful": false, "articles": []}
```

---

## Building and Pushing to Docker Hub

```bash
# Build
docker build -t yourusername/ccpa-compliance:latest .

# Test locally
docker run -p 8000:8000 yourusername/ccpa-compliance:latest

# In another terminal
python validate_format.py

# Push
docker login
docker push yourusername/ccpa-compliance:latest
```

---

## Project Structure

```
ccpa_compliance/
├── app/
│   └── main.py              # FastAPI server — full 65-page CCPA statute embedded
├── Dockerfile               # Pre-downloads model at build time
├── docker-compose.yml       # Easy single-command startup
├── requirements.txt         # Python dependencies
├── validate_format.py       # Hackathon format checker
└── README.md                # This file
```

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| `google/flan-t5-base` | Free, public, ~250MB, no HF token, CPU-compatible, fast |
| Full statute embedded verbatim | LLM has exact legal text to reason against; no RAG complexity |
| Keyword layer first | Deterministic, fast, catches 90%+ of obvious violations |
| LLM as second layer | Catches nuanced / implicit violations keyword matching misses |
| Union of both layers | Maximum recall — never misses a violation either layer catches |
| Model pre-downloaded at build time | Eliminates slow first-request delays in production |
| Model loaded once at startup | Prevents per-request model loading latency |

