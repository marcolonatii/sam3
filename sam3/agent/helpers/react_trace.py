from IPython.display import Markdown, display


def _format_message_content(content):
    """Convert LangChain message content (str or multimodal blocks) to displayable text."""
    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block.strip())
                continue

            if not isinstance(block, dict):
                parts.append(str(block))
                continue

            block_type = block.get("type")
            if block_type == "text":
                parts.append(block.get("text", ""))
            elif block_type == "image_url":
                image_ref = block.get("image_url") or block.get("url")
                if isinstance(image_ref, dict):
                    image_ref = image_ref.get("url", "")
                if isinstance(image_ref, str) and image_ref.startswith("data:"):
                    image_ref = "data URL"
                parts.append(f"[image_url: {image_ref or 'unknown'}]")
            else:
                parts.append(str(block))

        return "\n".join(p for p in parts if p).strip()

    return str(content).strip()


def format_agent_trace(messages):
    trace_md = "## Agent Execution Trace\n\n"

    for i, msg in enumerate(messages):
        msg_type = type(msg).__name__
        content = _format_message_content(getattr(msg, "content", ""))

        if msg_type == "HumanMessage":
            trace_md += f"**🧑 User Query** (Step {i+1})\n\n{content}\n\n"

        elif msg_type == "AIMessage" and msg.tool_calls:
            tool_call = msg.tool_calls[0]
            trace_md += f"**🤖 LLM Thought & Tool Call** (Step {i+1})\n\n"
            trace_md += f"**Model:** {msg.response_metadata.get('model_name', 'unknown')}\n"
            trace_md += f"**Tool Called:** `{tool_call['name']}`\n"
            trace_md += f"**Args:** `{tool_call['args']}`\n\n"

        elif msg_type == "ToolMessage":
            trace_md += f"**🛠️ Tool Observation** (Step {i+1})\n\n"
            trace_md += f"**Tool:** {msg.name}\n"
            trace_md += f"**Result:** {content}\n\n"

        elif msg_type == "AIMessage" and not msg.tool_calls:
            trace_md += f"**✅ Final Answer** (Step {i+1})\n\n"
            trace_md += f"**Response:** {content}\n\n"

    display(Markdown(trace_md))

# Paste your result dict here
