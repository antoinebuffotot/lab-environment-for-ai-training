"""OpenLLMetry (Traceloop SDK) -> Dynatrace OTLP, enabled only on demand.

Deliberately minimal : only what the application itself decides to trace is sent.
  - the current activity : spans rag.chat (workflow), rag.retrieve and rag.generate (tasks)
  - the prompt           : the CURRENT prompt only (system + current user message, never the
                           conversation history), as gen_ai.* attributes on rag.generate so
                           Dynatrace shows it in AI Observability, plus token usage counts
Nothing else : no automatic instrumentation (Ollama, ChromaDB, HTTP...), no completion text,
no metrics, no logs.
"""
import logging
import os
from contextlib import contextmanager

# Forced (not setdefault) : these must not be re-enabled by a stray environment variable.
os.environ["TRACELOOP_TELEMETRY"] = "false"        # Traceloop's own usage statistics
os.environ["TRACELOOP_TRACE_CONTENT"] = "false"    # no automatic capture of prompts / completions
os.environ["TRACELOOP_METRICS_ENABLED"] = "false"  # traces only

log = logging.getLogger("telemetry")
_tracer = None
MAX_ATTR = 8000


def _truthy(value) -> bool:
    return str(value).strip().lower() in ("1", "true", "y", "yes", "on")


def init_telemetry() -> bool:
    """Initialise OpenLLMetry if enabled. Must be called before importing ollama / chromadb."""
    global _tracer
    if not _truthy(os.getenv("OPENLLMETRY_ENABLED", "false")):
        log.info("OpenLLMetry disabled")
        return False

    tenant = os.getenv("DT_TENANT", "").strip().strip("/")
    token = os.getenv("DT_API_TOKEN", "").strip()
    if not tenant or not token:
        log.warning("OpenLLMetry enabled but DT_TENANT / DT_API_TOKEN missing: telemetry disabled")
        return False

    try:
        from opentelemetry import trace
        from traceloop.sdk import Traceloop

        kwargs = dict(
            app_name=os.getenv("OTEL_SERVICE_NAME", "ai-lab-rag-api"),
            api_endpoint=f"https://{tenant}/api/v2/otlp",
            headers={"Authorization": f"Api-Token {token}"},
            disable_batch=False,
            resource_attributes={
                "deployment.environment": "ai-training",
                "host.name": os.getenv("VM_FQDN", "localhost"),
            },
        )
        try:
            # block every automatic instrumentation : only our own spans are emitted
            from traceloop.sdk.instruments import Instruments
            kwargs.update(instruments=set(Instruments), block_instruments=set(Instruments))
        except Exception:
            log.warning("cannot restrict automatic instrumentations with this SDK version")

        try:
            Traceloop.init(**kwargs)
        except TypeError:  # older SDK without instruments / block_instruments
            kwargs.pop("instruments", None)
            kwargs.pop("block_instruments", None)
            Traceloop.init(**kwargs)

        _tracer = trace.get_tracer("ai-lab")
        log.info("OpenLLMetry enabled (activity + prompt only) -> %s", kwargs["api_endpoint"])
        return True
    except Exception:  # never prevent the app from starting
        log.exception("OpenLLMetry initialisation failed: telemetry disabled")
        _tracer = None
        return False


def is_enabled() -> bool:
    return _tracer is not None


@contextmanager
def span(name: str, kind: str = "task", **attributes):
    """Activity span; `kind` is 'workflow' or 'task' (Traceloop semantic attribute)."""
    if _tracer is None:
        yield None
        return
    with _tracer.start_as_current_span(name) as current:
        current.set_attribute("traceloop.span.kind", kind)
        current.set_attribute("traceloop.entity.name", name)
        for key, value in attributes.items():
            if value is not None:
                current.set_attribute(key, value)
        yield current


def trace_prompt(current, messages: list[dict], model: str, params: dict) -> None:
    """Record the current prompt and the request settings on a span, using the GenAI semantic
    conventions that Dynatrace AI Observability reads (gen_ai.system, gen_ai.prompt.N.*, ...).

    `messages` must only hold the current prompt (system + current user message): the
    conversation history is never traced.
    """
    if current is None:
        return
    current.set_attribute("gen_ai.system", "Ollama")
    current.set_attribute("gen_ai.operation.name", "chat")
    current.set_attribute("llm.request.type", "chat")
    current.set_attribute("gen_ai.request.model", model)
    for key, value in params.items():
        current.set_attribute(f"gen_ai.request.{key}", value)
    for i, message in enumerate(messages):
        current.set_attribute(f"gen_ai.prompt.{i}.role", message["role"])
        current.set_attribute(f"gen_ai.prompt.{i}.content", str(message["content"])[:MAX_ATTR])


def trace_usage(current, model: str, input_tokens: int, output_tokens: int) -> None:
    """Record token usage (numbers only, no completion text) for AI Observability cost/usage views."""
    if current is None:
        return
    current.set_attribute("gen_ai.response.model", model)
    current.set_attribute("gen_ai.usage.input_tokens", int(input_tokens))
    current.set_attribute("gen_ai.usage.output_tokens", int(output_tokens))
    current.set_attribute("llm.usage.total_tokens", int(input_tokens) + int(output_tokens))
