"""Agentic AI : a tool-calling loop on the same local model (Ollama) as the chat.

The agent receives a task, decides which tools to call (document search, calculator, clock, document
list), reads the results and repeats until it can answer, within a step limit.

Traces (OpenLLMetry, see telemetry.py):
    agent.run                @agent   the whole task (input = the task, output = the final answer)
    ├── agent.step           @task    one model call (the Ollama LLM span nests inside)
    ├── search_documents     @tool    one span per tool call
    └── ...
Plain async functions only (no generators) so every span starts and ends cleanly; the events shown
in the web UI reach the HTTP response through a queue.
"""
import ast
import asyncio
import json
import logging
import operator
import time
from datetime import datetime, timezone

from app import mcp_tools, rag
from app.rag import field
from app.telemetry import HISTORY_PREFIX, agent, task, tool

log = logging.getLogger("agent")

AGENT_SYSTEM_PROMPT = (
    "Tu es un agent IA autonome dans un atelier de formation sur l'IA générative et l'observabilité. "
    "Pour accomplir la tâche de l'utilisateur, tu disposes d'outils : utilise-les dès qu'ils apportent "
    "une information fiable (documents de l'atelier, calcul exact, date et heure) au lieu de deviner. "
    "D'autres outils (par exemple pour interroger une plateforme Dynatrace) peuvent s'ajouter à ceux-ci. "
    "Tu peux appeler plusieurs outils à la suite. Quand tu as assez d'éléments, réponds directement, "
    "de façon claire et concise, dans la langue de la tâche, et cite les documents utilisés "
    "(nom du fichier) quand tu t'appuies dessus."
)

DEFAULT_AGENT_PARAMS = {"temperature": 0.2, "top_k": 40, "top_p": 0.9, "max_steps": 5}
MAX_RESULT_CHARS = 3000

TOOLS = [
    {"type": "function", "function": {
        "name": "search_documents",
        "description": "Cherche dans les documents de l'atelier (base de connaissances) les passages les plus "
                       "pertinents pour une requête. À utiliser pour toute question sur l'atelier, OpenLLMetry, "
                       "le RAG ou l'infrastructure.",
        "parameters": {"type": "object", "required": ["query"], "properties": {
            "query": {"type": "string", "description": "La requête de recherche, en langage naturel"},
            "passages": {"type": "integer", "description": "Nombre de passages à retourner (1 à 5, défaut 3)"}}}}},
    {"type": "function", "function": {
        "name": "list_documents",
        "description": "Liste les documents indexés dans la base de connaissances.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "calculator",
        "description": "Calcule exactement une expression arithmétique (+ - * / // % ** et parenthèses).",
        "parameters": {"type": "object", "required": ["expression"], "properties": {
            "expression": {"type": "string", "description": "Par exemple : (12.5 + 7) * 3"}}}}},
    {"type": "function", "function": {
        "name": "get_current_time",
        "description": "Donne la date et l'heure actuelles (UTC).",
        "parameters": {"type": "object", "properties": {}}}},
]


# --------------------------------------------------------------------------- tools

_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow,
    ast.USub: operator.neg, ast.UAdd: operator.pos,
}


def _eval(node):
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise ValueError("exposant trop grand")
        return _OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    raise ValueError("expression non autorisée")


@tool("search_documents")
async def tool_search_documents(query: str, passages: int = 3) -> str:
    sources = await rag.search(query, max(1, min(int(passages), 5)))
    if not sources:
        return "Aucun passage trouvé."
    return "\n\n".join(f"[{s['n']}] (source : {s['source']}, pertinence {round(s['score'] * 100)} %)\n{s['text']}"
                       for s in sources)


@tool("list_documents")
async def tool_list_documents() -> str:
    docs = await rag.list_documents()
    return "\n".join(f"- {d['source']} ({d['chunks']} passages)" for d in docs) or "Aucun document indexé."


@tool("calculator")
def tool_calculator(expression: str) -> str:
    try:
        return str(_eval(ast.parse(str(expression).strip(), mode="eval")))
    except ZeroDivisionError:
        return "Erreur : division par zéro."
    except Exception as exc:
        return f"Erreur : {exc}"


@tool("get_current_time")
def tool_get_current_time() -> str:
    return datetime.now(timezone.utc).strftime("%A %d %B %Y, %H:%M:%S UTC")


async def run_mcp_tool(name: str, args: dict) -> str:
    """One span per MCP tool call, named after the tool (the decorator is built at call time)."""
    @tool(name)
    async def call(arguments: dict) -> str:
        return await mcp_tools.call(name, arguments)

    return await call(args)


async def run_tool(name: str, args: dict) -> str:
    """Execute one tool call; errors are returned to the model as text, never raised."""
    try:
        if name == "search_documents":
            result = await tool_search_documents(str(args.get("query", "")), args.get("passages", 3))
        elif name == "list_documents":
            result = await tool_list_documents()
        elif name == "calculator":
            result = tool_calculator(str(args.get("expression", "")))
        elif name == "get_current_time":
            result = tool_get_current_time()
        elif mcp_tools.has_tool(name):
            result = await run_mcp_tool(name, args)
        else:
            return f"Erreur : outil inconnu « {name} »."
    except Exception as exc:
        log.exception("tool %s failed", name)
        return f"Erreur pendant l'exécution de l'outil : {exc}"
    return str(result)[:MAX_RESULT_CHARS]


# --------------------------------------------------------------------------- messages

def build_agent_messages(task_text: str, history: list[dict], system_prompt: str = "") -> list[dict]:
    """Same layout as rag.build_messages: the task first (first input shown in Dynatrace), then the
    system instruction, then the previous turns in a marked message that the trace callback removes."""
    messages = [
        {"role": "user", "content": task_text},
        {"role": "system", "content": system_prompt.strip() or AGENT_SYSTEM_PROMPT},
    ]
    previous = []
    for item in history[-6:]:
        role, content = item.get("role"), str(item.get("content", ""))[:3000]
        if role in ("user", "assistant") and content:
            previous.append(f"{'Utilisateur' if role == 'user' else 'Assistant'} : {content}")
    if previous:
        messages.append({"role": "user", "content": (
            f"{HISTORY_PREFIX}\n\n" + "\n\n".join(previous) + "\n\n"
            "Tiens compte de ces échanges pour accomplir la tâche ci-dessus.")})
    return messages


def _tool_call_dicts(calls) -> list[dict]:
    out = []
    for call in calls or []:
        fn = field(call, "function", {})
        args = field(fn, "arguments", {})
        out.append({"function": {"name": field(fn, "name", ""), "arguments": dict(args) if args else {}}})
    return out


# --------------------------------------------------------------------------- the agent

def agent_turn(req, queue: asyncio.Queue, sse):
    """Build the traced agent run for one request. Events go to `queue` as SSE strings."""
    totals = {"steps": 0, "tokens": 0, "tps": 0.0}

    @agent("agent.run")
    async def run(task_text: str) -> str:
        messages = build_agent_messages(task_text, req.history, req.system_prompt)
        sampling = {"temperature": req.temperature, "top_k": req.top_k, "top_p": req.top_p, "num_ctx": 4096}
        final = ""
        await mcp_tools.refresh()             # picks up the mcp.json created or edited by the participant
        all_tools = TOOLS + mcp_tools.tool_schemas()

        for step in range(1, req.max_steps + 1):
            use_tools = step < req.max_steps      # last step : no tools, the model must answer
            await queue.put(sse("step", {"n": step}))

            @task("agent.step")
            async def think() -> dict:
                content, calls = [], []
                stream = await rag._client.chat(
                    model=rag.LLM_MODEL, messages=messages, tools=all_tools if use_tools else None,
                    stream=True, options=sampling)
                async for part in stream:
                    msg = field(part, "message", {})
                    text = field(msg, "content", "")
                    if text:
                        content.append(text)
                        await queue.put(sse("token", {"text": text}))
                    calls.extend(field(msg, "tool_calls", []) or [])
                    if field(part, "done", False):
                        n = field(part, "eval_count", 0)
                        duration = field(part, "eval_duration", 0) / 1e9
                        totals["tokens"] += n
                        if duration:
                            totals["tps"] = round(n / duration, 2)
                return {"content": "".join(content), "tool_calls": _tool_call_dicts(calls)}

            reply = await think()
            totals["steps"] = step
            if not reply["tool_calls"]:
                final = reply["content"]
                break

            messages.append({"role": "assistant", "content": reply["content"], "tool_calls": reply["tool_calls"]})
            for call in reply["tool_calls"]:
                name, args = call["function"]["name"], call["function"]["arguments"]
                await queue.put(sse("tool_call", {"name": name, "args": args}))
                result = await run_tool(name, args)
                await queue.put(sse("tool_result", {"name": name, "result": result}))
                messages.append({"role": "tool", "tool_name": name, "content": result})
        return final

    return run, totals
