"""Agentic AI with LangChain : the same agent as agent.py, built with LangChain and LangGraph.

Selected with AGENT_FRAMEWORK=langchain (default: python, the hand-written loop of agent.py). Everything the participants
see is identical: same page, same tools (local tools, Dynatrace sub-agent, MCP, dtctl, skills), same SSE events. Only the
engine changes:

    ChatOllama                      the model (langchain-ollama), same qwen2.5:7b served by Ollama
    StructuredTool                  every tool of agent.py, exposed to LangChain with its JSON schema
    create_agent (LangGraph)        the tool-calling loop (model node <-> tools node)
    graph.astream(stream_mode=...)  tokens and node updates, translated into the SSE events of the web UI

Traces (OpenLLMetry): here the spans come from the LangChain instrumentation of the SDK (workflow, model calls and tool
calls are traced automatically, with the run names below), not from the @agent/@task/@tool decorators of agent.py.
    agent.run                       the main agent (LangGraph run)
    ├── <chat model calls>          one span per model call
    ├── <tool>                      one span per tool call (calculator, search_documents, ...)
    └── ask_dynatrace               delegation to the sub-agent
        └── dynatrace_agent         its own model calls and tools
"""
import asyncio
import logging

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langchain_ollama import ChatOllama
from langgraph.errors import GraphRecursionError

from app import mcp_tools, rag
from app.agent import (ASK_DYNATRACE, DT_MAX_STEPS, NUM_CTX, TOOLS, build_agent_messages, dt_system_prompt,
                       dynatrace_tool_schemas, run_tool, subagent_active)
from app.limits import clip

log = logging.getLogger("agent-langchain")

SUB_TAG = "dynatrace-subagent"     # tag of the sub-agent runs: the main agent ignores the (inherited) events carrying it
FINAL_PROMPT = "Use what you have gathered so far to give your final answer now. Do not call any tool."


def make_llm(req) -> ChatOllama:
    """The chat model: same Ollama server, same model and same sampling settings as the hand-written agent."""
    return ChatOllama(model=rag.LLM_MODEL, base_url=rag.OLLAMA_HOST, temperature=req.temperature,
                      top_k=req.top_k, top_p=req.top_p, num_ctx=NUM_CTX)


def _text(content) -> str:
    """Text of a LangChain message content (a string, or a list of content blocks)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(b if isinstance(b, str) else (b.get("text") or "") for b in content if isinstance(b, (str, dict)))
    return str(content or "")


def lc_tools(schemas: list[dict], runner) -> list[StructuredTool]:
    """LangChain tools for the OpenAI-style function schemas of agent.py; the call goes through the same run_tool."""
    out = []
    for s in schemas:
        f = s["function"]

        async def call(_name=f["name"], **kwargs) -> str:
            return await runner(_name, kwargs)

        out.append(StructuredTool(name=f["name"], description=f["description"], args_schema=f["parameters"],
                                  coroutine=call))
    return out


def agent_turn(req, queue: asyncio.Queue, sse):
    """Build the run of one request with LangChain. Same contract as agent.agent_turn: returns (run, totals)."""
    totals = {"steps": 0, "tokens": 0, "tps": 0.0}
    llm = make_llm(req)

    async def drive(graph, messages: list, who: str, max_steps: int, run_name: str, tags: list[str]) -> str:
        """Run a LangGraph agent and translate its stream into the SSE events of the web UI; returns the final answer."""
        step, last_graph_step = 0, None
        transcript, answer = list(messages), ""
        # max_steps model calls in total, the last one without tools: (max_steps - 1) model+tools rounds, then the
        # step limit is hit and the final answer is requested without tools (see GraphRecursionError below)
        config = {"recursion_limit": max(1, 2 * (max_steps - 1)), "run_name": run_name, "tags": tags}

        async def emit(event: str, data: dict):
            await queue.put(sse(event, {**data, "agent": who}))

        async def new_step():
            nonlocal step
            step += 1
            totals["steps"] += 1
            await emit("step", {"n": step})

        try:
            async for mode, data in graph.astream({"messages": messages}, config=config,
                                                  stream_mode=["messages", "updates"]):
                if mode == "messages":
                    chunk, meta = data
                    if not isinstance(chunk, AIMessageChunk) or meta.get("langgraph_node") != "model":
                        continue
                    if who == "main" and SUB_TAG in (meta.get("tags") or []):
                        continue                       # tokens of the sub-agent, inherited through the tool call
                    if meta.get("langgraph_step") != last_graph_step:
                        last_graph_step = meta.get("langgraph_step")
                        await new_step()
                    text = _text(chunk.content)
                    if text:
                        await emit("token", {"text": text})
                    continue
                for node, update in (data or {}).items():       # mode == "updates"
                    for msg in (update or {}).get("messages", []) if isinstance(update, dict) else []:
                        transcript.append(msg)
                        if isinstance(msg, ToolMessage):
                            await emit("tool_result", {"id": msg.tool_call_id, "name": msg.name or "",
                                                       "result": _text(msg.content)})
                        elif isinstance(msg, AIMessage):
                            await report_usage(msg, emit)
                            if msg.tool_calls:
                                for call in msg.tool_calls:
                                    await emit("tool_call", {"id": call.get("id"), "name": call["name"],
                                                             "args": call.get("args") or {}})
                            else:
                                answer = _text(msg.content)
        except GraphRecursionError:
            # step limit reached: like the hand-written agent, the model must now answer without tools
            await new_step()
            parts = []
            async for chunk in llm.astream(transcript + [HumanMessage(FINAL_PROMPT)], config={"tags": tags}):
                text = _text(chunk.content)
                if text:
                    parts.append(text)
                    await emit("token", {"text": text})
                if chunk.usage_metadata:
                    totals["tokens"] += int(chunk.usage_metadata.get("output_tokens") or 0)
            answer = "".join(parts)
        return answer

    async def report_usage(msg: AIMessage, emit) -> None:
        """Token counters of the run, and the same context warnings as the hand-written agent."""
        meta = msg.response_metadata or {}
        usage = msg.usage_metadata or {}
        log.debug("model call usage=%s metadata=%s", usage, meta)
        totals["tokens"] += int(usage.get("output_tokens") or meta.get("eval_count") or 0)
        duration = (meta.get("eval_duration") or 0) / 1e9
        if duration and meta.get("eval_count"):
            totals["tps"] = round(meta["eval_count"] / duration, 2)
        used_prompt = int(usage.get("input_tokens") or meta.get("prompt_eval_count") or 0)
        if meta.get("done_reason") == "length":
            await emit("warning", {"code": "length", "ctx": NUM_CTX})
        elif used_prompt >= 0.9 * NUM_CTX:
            await emit("warning", {"code": "context", "tokens": used_prompt, "ctx": NUM_CTX})

    async def ask_dynatrace(question: str) -> str:
        """Sub-agent: its own LangGraph agent, its own instruction and context, only the Dynatrace tools."""
        sub = create_agent(llm, lc_tools(dynatrace_tool_schemas(), run_tool))
        messages = [HumanMessage(question), SystemMessage(dt_system_prompt())]
        answer = await drive(sub, messages, "dynatrace", DT_MAX_STEPS, "dynatrace_agent", [SUB_TAG])
        return clip(answer or "Le sous-agent Dynatrace n'a pas produit de réponse.")

    async def main_tool(name: str, args: dict) -> str:
        if name == "ask_dynatrace":
            try:
                return await ask_dynatrace(str(args.get("question", "")))
            except Exception as exc:
                log.exception("dynatrace sub-agent failed")
                return f"Erreur du sous-agent Dynatrace : {exc}"
        return await run_tool(name, args)

    async def run(task_text: str) -> str:
        await mcp_tools.refresh()             # picks up the mcp.json created or edited by the participant
        schemas = TOOLS + ([ASK_DYNATRACE] if subagent_active() else dynatrace_tool_schemas())
        graph = create_agent(llm, lc_tools(schemas, main_tool))
        messages = [_to_lc(m) for m in build_agent_messages(task_text, req.history, req.system_prompt)]
        return await drive(graph, messages, "main", req.max_steps, "agent.run", [])

    return run, totals


def _to_lc(message: dict):
    """The {role, content} messages built by agent.build_agent_messages, as LangChain messages."""
    if message["role"] == "system":
        return SystemMessage(message["content"])
    return HumanMessage(message["content"])
