"""OpenLLMetry (Traceloop SDK) -> Dynatrace AI Observability, enabled only on demand.

Follows https://docs.dynatrace.com/docs/observe/dynatrace-for-ai-observability/get-started/openllmetry :
  - Traceloop.init(...) pointed at the tenant OTLP endpoint (disable_batch=True),
  - metrics temporality set to "delta",
  - @workflow / @task decorators for the application activity (rag.chat, rag.retrieve, rag.generate).
The LLM span (model, parameters, token usage, prompt and completion) is produced by the SDK's own
Ollama instrumentation: nothing is hand-written. Other instrumentations (ChromaDB, HTTP...) are blocked.
"""
import logging
import os
import re

# Forced (not setdefault) : these must not be changed by a stray environment variable.
os.environ["TRACELOOP_TELEMETRY"] = "false"                          # Traceloop's own usage statistics
os.environ["OTEL_EXPORTER_OTLP_METRICS_TEMPORALITY_PREFERENCE"] = "delta"   # required by Dynatrace

log = logging.getLogger("telemetry")
_enabled = False


def _truthy(value) -> bool:
    return str(value).strip().lower() in ("1", "true", "y", "yes", "on")


def _auth_header(token: str) -> dict:
    # classic access tokens (dt0c01.*) use "Api-Token", platform tokens (dt0s16.*) use "Bearer"
    scheme = "Api-Token" if token.startswith("dt0c01.") else "Bearer"
    return {"Authorization": f"{scheme} {token}"}


def service_name() -> str:
    """Service name suffixed with the VM number taken from the host name.

    aitraining03.francecentral.cloudapp.azure.com -> ai-lab-rag-api-03 (no suffix if no digits).
    """
    base = os.getenv("OTEL_SERVICE_NAME", "ai-lab-rag-api")
    first_label = os.getenv("VM_FQDN", "").split(".")[0]
    number = re.search(r"(\d+)$", first_label)
    return f"{base}-{number.group(1)}" if number else base


def init_telemetry() -> bool:
    """Initialise OpenLLMetry if enabled. Must be called before importing ollama / chromadb."""
    global _enabled
    if not _truthy(os.getenv("OPENLLMETRY_ENABLED", "false")):
        log.info("OpenLLMetry disabled")
        return False

    tenant = os.getenv("DT_TENANT", "").strip().strip("/")
    token = os.getenv("DT_API_TOKEN", "").strip()
    if not tenant or not token:
        log.warning("OpenLLMetry enabled but DT_TENANT / DT_API_TOKEN missing: telemetry disabled")
        return False

    # prompts and completions are recorded by the SDK unless explicitly turned off
    os.environ["TRACELOOP_TRACE_CONTENT"] = "true" if _truthy(os.getenv("OPENLLMETRY_TRACE_CONTENT", "true")) else "false"

    try:
        from traceloop.sdk import Traceloop

        kwargs = dict(
            app_name=service_name(),
            api_endpoint=f"https://{tenant}/api/v2/otlp",
            headers=_auth_header(token),
            disable_batch=True,
            resource_attributes={
                "deployment.environment": "ai-training",
                "host.name": os.getenv("VM_FQDN", "localhost"),
            },
        )
        try:
            # only the Ollama (LLM) instrumentation: block everything else
            from traceloop.sdk.instruments import Instruments
            kwargs.update(instruments={Instruments.OLLAMA})
        except Exception:
            log.warning("cannot restrict automatic instrumentations with this SDK version")

        try:
            Traceloop.init(**kwargs)
        except TypeError:  # older SDK without `instruments`
            kwargs.pop("instruments", None)
            Traceloop.init(**kwargs)

        _enabled = True
        log.info("OpenLLMetry enabled: service %s -> %s (trace content: %s)", kwargs["app_name"],
                 kwargs["api_endpoint"], os.environ["TRACELOOP_TRACE_CONTENT"])
        return True
    except Exception:  # never prevent the app from starting
        log.exception("OpenLLMetry initialisation failed: telemetry disabled")
        _enabled = False
        return False


def is_enabled() -> bool:
    return _enabled


def _decorator(kind: str, name: str):
    """Traceloop @workflow / @task when enabled, otherwise a no-op decorator."""
    if not _enabled:
        return lambda fn: fn
    from traceloop.sdk import decorators
    return getattr(decorators, kind)(name=name)


def workflow(name: str):
    return _decorator("workflow", name)


def task(name: str):
    return _decorator("task", name)
