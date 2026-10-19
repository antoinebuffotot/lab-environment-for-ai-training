"""RAG API : chat (SSE streaming), search, document management."""
import asyncio
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

# OpenLLMetry must be initialised BEFORE ollama / chromadb are imported (see rag.py)
from app.telemetry import init_telemetry, is_enabled, span, trace_prompt  # noqa: E402

init_telemetry()

from fastapi import FastAPI, File, HTTPException, UploadFile  # noqa: E402
from fastapi.responses import StreamingResponse  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from app import rag  # noqa: E402

log = logging.getLogger("api")
DOCS_DIR = os.getenv("DOCS_DIR", "/data/docs")
MAX_UPLOAD = 5_000_000

state = {"models_ready": False, "bootstrap": "waiting_models", "error": ""}


async def bootstrap():
    """Wait for the models to be pulled, then index the sample documents once."""
    while True:
        try:
            if await rag.models_ready():
                break
        except Exception as exc:  # ollama not up yet
            state["error"] = str(exc)
        await asyncio.sleep(5)
    state["models_ready"] = True
    state["bootstrap"] = "ingesting"
    try:
        if await rag.count_chunks() == 0:
            await rag.ingest_dir(DOCS_DIR)
        state["bootstrap"] = "done"
        state["error"] = ""
    except Exception as exc:
        log.exception("bootstrap failed")
        state["bootstrap"] = "error"
        state["error"] = str(exc)


@asynccontextmanager
async def lifespan(_app):
    task = asyncio.create_task(bootstrap())
    yield
    task.cancel()


app = FastAPI(title="AI lab RAG API", lifespan=lifespan)


D = rag.DEFAULT_PARAMS


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    history: list[dict] = []
    use_rag: bool = True
    passages: int = Field(D["passages"], ge=1, le=8)            # RAG : number of passages retrieved
    system_prompt: str = Field("", max_length=4000)             # empty = default system message
    temperature: float = Field(D["temperature"], ge=0, le=2)    # sampling
    top_k: int = Field(D["top_k"], ge=1, le=100)                # sampling (token candidates)
    top_p: float = Field(D["top_p"], ge=0, le=1)                # sampling (nucleus)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    passages: int = Field(D["passages"], ge=1, le=8)


def sse(event: str, data) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.get("/api/health")
async def health():
    chunks = 0
    if state["bootstrap"] == "done":
        try:
            chunks = await rag.count_chunks()
        except Exception:
            pass
    return {
        "ready": state["models_ready"] and state["bootstrap"] == "done",
        "models_ready": state["models_ready"],
        "bootstrap": state["bootstrap"],
        "error": state["error"],
        "chunks": chunks,
        "model": rag.LLM_MODEL,
        "embed_model": rag.EMBED_MODEL,
        "openllmetry": is_enabled(),
        "host": os.getenv("VM_FQDN", "localhost"),
    }


@app.get("/api/settings")
async def settings():
    """Default chat settings, used by the web UI to fill and reset the settings panel."""
    return {**D, "system_prompt": rag.SYSTEM_PROMPT}


@app.post("/api/chat")
async def chat(req: ChatRequest):
    if not state["models_ready"]:
        raise HTTPException(503, "Le modèle est encore en cours de téléchargement.")

    async def stream():
        started = time.time()
        try:
            with span("rag.chat", kind="workflow", **{"rag.use_retrieval": req.use_rag}):
                sources: list[dict] = []
                if req.use_rag:
                    with span("rag.retrieve", kind="task", **{"rag.passages": req.passages}):
                        sources = await rag.search(req.message, req.passages)
                    yield sse("sources", sources)

                messages = rag.build_messages(req.message, req.history, sources, req.system_prompt)
                sampling = {"temperature": req.temperature, "top_k": req.top_k, "top_p": req.top_p}
                with span("rag.generate", kind="task", **{"rag.sources": len(sources)}) as gen:
                    trace_prompt(gen, messages, rag.LLM_MODEL, sampling)   # the only content traced
                    tokens, tps = 0, 0.0
                    async for part in await rag.chat_stream(messages, **sampling):
                        text = rag.field(rag.field(part, "message", {}), "content", "")
                        if text:
                            yield sse("token", {"text": text})
                        if rag.field(part, "done", False):
                            tokens = rag.field(part, "eval_count", 0)
                            duration = rag.field(part, "eval_duration", 0) / 1e9
                            tps = round(tokens / duration, 2) if duration else 0.0
                yield sse("done", {"tokens": tokens, "tps": tps, "total_s": round(time.time() - started, 1)})
        except Exception as exc:
            log.exception("chat failed")
            yield sse("error", {"text": f"Erreur pendant la génération : {exc}"})

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/search")
async def search(req: SearchRequest):
    if not state["models_ready"]:
        raise HTTPException(503, "Le modèle d'embedding n'est pas encore prêt.")
    return {"sources": await rag.search(req.query, req.passages)}


@app.get("/api/documents")
async def documents():
    if state["bootstrap"] != "done":
        return []
    return await rag.list_documents()


@app.delete("/api/documents/{source}")
async def delete_document(source: str):
    await rag.delete_document(Path(source).name)
    return {"deleted": source}


@app.post("/api/ingest")
async def ingest():
    if not state["models_ready"]:
        raise HTTPException(503, "Le modèle d'embedding n'est pas encore prêt.")
    return {"indexed": await rag.ingest_dir(DOCS_DIR)}


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    if not state["models_ready"]:
        raise HTTPException(503, "Le modèle d'embedding n'est pas encore prêt.")
    name = Path(file.filename or "document.txt").name
    if Path(name).suffix.lower() not in rag.ALLOWED_EXT:
        raise HTTPException(415, "Formats acceptés : .txt, .md, .pdf")
    data = await file.read()
    if len(data) > MAX_UPLOAD:
        raise HTTPException(413, "Fichier trop volumineux (5 Mo maximum).")
    text = rag.extract_text(name, data)
    if not text.strip():
        raise HTTPException(422, "Aucun texte exploitable dans ce fichier.")
    return {"source": name, "chunks": await rag.ingest_text(name, text)}
