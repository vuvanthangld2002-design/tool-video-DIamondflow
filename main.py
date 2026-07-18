"""
Diamond Flow — FastAPI Backend
- Serve static + templates with CORS headers for FFmpeg.wasm
- Proxy API calls to Groq/Gemini (keeps API keys server-side)
"""

import os
import httpx
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

app = FastAPI(title="Diamond Flow Studio")

# ---------------------------------------------------------------------------
# CORS / COOP / COEP headers for FFmpeg.wasm (SharedArrayBuffer)
# ---------------------------------------------------------------------------
@app.middleware("http")
async def add_coop_coep_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
    response.headers["Cross-Origin-Embedder-Policy"] = "credentialless"
    return response

# ---------------------------------------------------------------------------
# Static files & templates (FIXED FOR VERCEL SERVERLESS)
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Ép hệ thống tự tạo thư mục nếu Vercel lỡ "bỏ quên" lúc đóng gói
static_dir = os.path.join(BASE_DIR, "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

templates_dir = os.path.join(BASE_DIR, "templates")
os.makedirs(templates_dir, exist_ok=True)
templates = Jinja2Templates(directory=templates_dir)

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

# ---------------------------------------------------------------------------
# API Key storage
# ---------------------------------------------------------------------------
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

# ---------------------------------------------------------------------------
# Proxy: Groq Chat Completions
# ---------------------------------------------------------------------------
class GroqRequest(BaseModel):
    api_key: str | None = None  # optional override
    model: str
    messages: list[dict]
    temperature: float = 0.85
    max_tokens: int = 8192
    response_format: dict | None = None

@app.post("/api/groq")
async def proxy_groq(body: GroqRequest):
    api_key = body.api_key or GROQ_API_KEY
    if not api_key:
        return JSONResponse({"error": "No Groq API key provided"}, status_code=400)

    payload = {
        "model": body.model,
        "messages": body.messages,
        "temperature": body.temperature,
        "max_tokens": body.max_tokens,
    }
    if body.response_format:
        payload["response_format"] = body.response_format

    async with httpx.AsyncClient(timeout=120) as client:
        try:
            res = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}",
                },
                json=payload,
            )
            return JSONResponse(status_code=res.status_code, content=res.json())
        except httpx.TimeoutException:
            return JSONResponse({"error": "Groq API timeout"}, status_code=504)
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)

# ---------------------------------------------------------------------------
# Proxy: Gemini generateContent
# ---------------------------------------------------------------------------
class GeminiRequest(BaseModel):
    api_key: str | None = None
    model: str
    contents: list[dict]
    generation_config: dict | None = None

@app.post("/api/gemini")
async def proxy_gemini(body: GeminiRequest):
    api_key = body.api_key or GEMINI_API_KEY
    if not api_key:
        return JSONResponse({"error": "No Gemini API key provided"}, status_code=400)

    payload = {"contents": body.contents}
    if body.generation_config:
        payload["generationConfig"] = body.generation_config

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{body.model}:generateContent?key={api_key}"

    async with httpx.AsyncClient(timeout=120) as client:
        try:
            res = await client.post(url, json=payload, headers={"Content-Type": "application/json"})
            return JSONResponse(status_code=res.status_code, content=res.json())
        except httpx.TimeoutException:
            return JSONResponse({"error": "Gemini API timeout"}, status_code=504)
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
