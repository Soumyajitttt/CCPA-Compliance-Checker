CCPA Compliance Checker

OPEN HACK 2026 — CSA, IISc

1️⃣ Solution Overview
Objective

This system analyzes natural-language descriptions of business data practices and determines whether the practice violates the California Consumer Privacy Act (CCPA). If a violation is detected, the system returns the exact statute section(s) violated in a strictly formatted JSON response.

Technical Approach

This implementation uses a deterministic rule-based compliance engine built on structured regular expressions.

Instead of relying on a large language model, this system uses:

FastAPI — HTTP server

Pydantic — request validation

Python re module — structured legal rule matching

Docker — containerization

System Architecture
Client Request
      ↓
FastAPI Server (/analyze)
      ↓
Input Normalization
      ↓
Rule Engine Evaluation
      ↓
Section Matching
      ↓
Strict JSON Response
Processing Pipeline (End-to-End)
Step 1 — Input

The system receives:

{"prompt": "<natural language business practice>"}
Step 2 — Normalization

The prompt is:

Lowercased

Whitespace-normalized

Cleaned for consistent matching

Step 3 — Rule Evaluation

Each CCPA section is encoded as a structured rule object containing:

strong_any — Direct violation indicators

all_of — Required keyword combinations

proximity — Distance-based keyword matching

suppress_if_any — Compliance language overrides

Step 4 — Section Identification

If a rule matches, its corresponding statute section is added to the result list:

Example:

"Section 1798.120"

Multiple violations are supported.

Step 5 — Strict JSON Output

The system returns:

{
  "harmful": true | false,
  "articles": ["Section 1798.xxx", ...]
}

Rules enforced:

harmful is a boolean (not a string)

articles is always a list

If harmful = false → articles = []

If harmful = true → articles must be non-empty

No extra text, markdown, or explanation

2️⃣ Docker Run Command (MANDATORY)
Pull from Docker Hub
docker pull yourusername/ccpa-compliance:latest
Run the container
docker run -p 8000:8000 yourusername/ccpa-compliance:latest

The server will be available at:

http://localhost:8000
3️⃣ Environment Variables

This implementation does not require any environment variables.

Variable	Required	Description
None	No	No tokens, API keys, or external services required

No HF_TOKEN

No external APIs

No model downloads

4️⃣ GPU Requirements

This system:

Does NOT require a GPU

Runs fully on CPU

Has no CUDA dependency

Has no VRAM requirement

Minimum Requirements

1 CPU core

~100MB RAM

Python 3.10+

CPU-only fallback

Fully supported (GPU not needed).

5️⃣ Local Setup Instructions (Fallback)

⚠ This section is used only if Docker deployment fails.

Requirements

Ubuntu 20.04+ (or any Linux VM)

Python 3.10+

pip

Step 1 — Install dependencies
pip install -r requirements.txt
Step 2 — Start FastAPI server
uvicorn main:app --host 0.0.0.0 --port 8000
Step 3 — Verify server
curl http://localhost:8000/health

Expected response:

{"status": "ok"}
6️⃣ API Usage Examples (MANDATORY)
GET /health
curl http://localhost:8000/health

Response:

{"status": "ok"}
POST /analyze
Example 1 — Violation
curl -X POST http://localhost:8000/analyze \
     -H "Content-Type: application/json" \
     -d '{"prompt":"We sell personal data without offering opt-out."}'

Response:

{
  "harmful": true,
  "articles": ["Section 1798.120"]
}
Example 2 — Compliant Practice
curl -X POST http://localhost:8000/analyze \
     -H "Content-Type: application/json" \
     -d '{"prompt":"We provide a clear privacy policy and honor deletion requests."}'

Response:

{
  "harmful": false,
  "articles": []
}
7️⃣ CCPA Sections Covered

The system includes rule coverage for:

Section 1798.100 — Notice at collection

Section 1798.105 — Right to deletion

Section 1798.110 — Right to know

Section 1798.115 — Disclosure of third parties

Section 1798.120 — Sale of personal information

Section 1798.121 — Sensitive personal information

Section 1798.125 — Discrimination

Section 1798.130 — Consumer request methods

Section 1798.135 — Do Not Sell link

Section 1798.150 — Data breach liability

8️⃣ Compliance with Hackathon Evaluation

This system satisfies all technical requirements:

✔ Listens on port 8000
✔ Exposes GET /health
✔ Exposes POST /analyze
✔ Returns strictly valid JSON
✔ No interactive prompts
✔ No model downloads at runtime
✔ No external dependencies
✔ Fast startup
✔ Deterministic output

9️⃣ Design Strengths

Deterministic (no hallucinated articles)

Fully auditable legal logic

Structured statute mapping

Suppression rules to reduce false positives

Extremely low latency

Docker-stable deployment

Minimal memory footprint

No GPU dependency

🔐 Hugging Face Token

Not required.

This implementation does not use a gated model and does not require an HF token.

Final Notes

This system was designed for:

Accuracy

Stability

Clean API behavior

Transparent compliance logic

Reliable Docker deployment

Strict adherence to evaluation rules
