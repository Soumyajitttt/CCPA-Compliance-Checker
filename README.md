# CCPA Compliance Checker

> **OPEN HACK 2026 — CSA, IISc**  
> Analyzes natural-language descriptions of business data practices and determines CCPA compliance.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Quick Start (Docker)](#quick-start-docker)
- [Local Setup (Fallback)](#local-setup-fallback)
- [API Reference](#api-reference)
- [CCPA Sections Covered](#ccpa-sections-covered)
- [System Requirements](#system-requirements)
- [Environment Variables](#environment-variables)
- [Evaluation Checklist](#evaluation-checklist)

---

## Overview

This system uses a **deterministic rule-based compliance engine** built on structured regular expressions to evaluate business data practices against the California Consumer Privacy Act (CCPA).

**Tech Stack:**
- **FastAPI** — HTTP server
- **Pydantic** — Request validation
- **Python `re` module** — Structured legal rule matching
- **Docker** — Containerization

---

## Architecture
```
Client Request
      ↓
FastAPI Server (/analyze)
      ↓
Input Normalization (lowercase, whitespace-clean)
      ↓
Rule Engine Evaluation
      ↓
Section Matching
      ↓
Strict JSON Response
```

Each CCPA rule is encoded with:

| Field | Description |
|---|---|
| `strong_any` | Direct violation indicators |
| `all_of` | Required keyword combinations |
| `proximity` | Distance-based keyword matching |
| `suppress_if_any` | Compliance language overrides (reduces false positives) |

---

## Quick Start (Docker)

**Pull the image:**
```bash
docker pull yourusername/ccpa-compliance:latest
```

**Run the container:**
```bash
docker run -p 8000:8000 yourusername/ccpa-compliance:latest
```

Server available at: `http://localhost:8000`

---

## Local Setup (Fallback)

> Use only if Docker deployment fails.

**Requirements:** Ubuntu 20.04+, Python 3.10+, pip

**Step 1 — Install dependencies:**
```bash
pip install -r requirements.txt
```

**Step 2 — Start the server:**
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

**Step 3 — Verify:**
```bash
curl http://localhost:8000/health
# Expected: {"status": "ok"}
```

---

## API Reference

### `GET /health`
```bash
curl http://localhost:8000/health
```

**Response:**
```json
{"status": "ok"}
```

---

### `POST /analyze`

**Request body:**
```json
{"prompt": "<natural language business practice>"}
```

**Response schema:**
```json
{
  "harmful": true | false,
  "articles": ["Section 1798.xxx", ...]
}
```

**Response rules:**
- `harmful` is always a boolean (never a string)
- `articles` is always a list
- If `harmful = false` → `articles = []`
- If `harmful = true` → `articles` is non-empty
- No extra text, markdown, or explanation in response

---

#### Example 1 — Violation Detected
```bash
curl -X POST http://localhost:8000/analyze \
     -H "Content-Type: application/json" \
     -d '{"prompt":"We sell personal data without offering opt-out."}'
```
```json
{
  "harmful": true,
  "articles": ["Section 1798.120"]
}
```

#### Example 2 — Compliant Practice
```bash
curl -X POST http://localhost:8000/analyze \
     -H "Content-Type: application/json" \
     -d '{"prompt":"We provide a clear privacy policy and honor deletion requests."}'
```
```json
{
  "harmful": false,
  "articles": []
}
```

---

## CCPA Sections Covered

| Section | Description |
|---|---|
| 1798.100 | Notice at collection |
| 1798.105 | Right to deletion |
| 1798.110 | Right to know |
| 1798.115 | Disclosure of third parties |
| 1798.120 | Sale of personal information |
| 1798.121 | Sensitive personal information |
| 1798.125 | Non-discrimination |
| 1798.130 | Consumer request methods |
| 1798.135 | Do Not Sell link |
| 1798.150 | Data breach liability |

---

## System Requirements

| Resource | Requirement |
|---|---|
| CPU | 1 core (minimum) |
| RAM | ~100 MB |
| GPU | Not required |
| CUDA | Not required |
| Python | 3.10+ |
| HF Token | Not required |

---

## Environment Variables

No environment variables required. This system has no external API dependencies and no model downloads at runtime.

---

## Evaluation Checklist

| Requirement | Status |
|---|---|
| Listens on port 8000 | ✅ |
| `GET /health` endpoint | ✅ |
| `POST /analyze` endpoint | ✅ |
| Strictly valid JSON output | ✅ |
| No interactive prompts | ✅ |
| No model downloads at runtime | ✅ |
| No external dependencies | ✅ |
| Fast startup | ✅ |
| Deterministic output | ✅ |
