import hashlib

from litellm.integrations.custom_logger import CustomLogger


SMG_ROUTING_HEADER = "X-SMG-Routing-Key"
SUPPORTED_CALL_TYPES = {
    "completion",
    "acompletion",
    "text_completion",
    "atext_completion",
    "responses",
    "aresponses",
}


def clean(value):
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        value = value[0] if value else None
    if isinstance(value, bytes):
        value = value.decode("utf-8", "ignore")
    text = str(value).strip()
    if text == "" or text.lower() in {"none", "null", "undefined"}:
        return None
    return text[:512]


def get_attr(obj, name):
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def first_named(*items):
    for source, value in items:
        value = clean(value)
        if value is not None:
            return value, source
    return None, None


def header(headers, name):
    return headers.get(name.lower())


def normalize_headers(raw_headers):
    if raw_headers is None:
        return {}
    return {str(k).lower(): v for k, v in dict(raw_headers).items()}


def infer_agent(headers):
    user_agent = clean(header(headers, "user-agent")) or ""
    if header(headers, "x-openwebui-chat-id"):
        return "open-webui", "infer:x-openwebui-chat-id"
    if (
        header(headers, "x-roo-task-id")
        or header(headers, "x-roo-root-task-id")
        or header(headers, "x-roo-code-task-id")
        or header(headers, "x-roo-code-session-id")
    ):
        return "roo-code", "infer:roo-headers"
    if (
        header(headers, "x-kilo-session")
        or header(headers, "x-kilocode-taskid")
        or header(headers, "x-kilocode-mode")
        or header(headers, "x-kilo-code-session-id")
        or header(headers, "x-kilo-session-id")
    ):
        return "kilo-code", "infer:kilo-headers"
    if header(headers, "x-opencode-client") == "pi":
        return "pi", "infer:x-opencode-client"
    if header(headers, "x-opencode-session"):
        return "opencode", "infer:x-opencode-session"
    if header(headers, "session_id") or header(headers, "x-client-request-id"):
        return "pi", "infer:pi-affinity-headers"
    if user_agent.startswith("claude-cli/"):
        return "claude-code", "infer:user-agent"
    for marker, agent in (
        ("kilo-code", "kilo-code"),
        ("kilocode", "kilo-code"),
        ("kilo code", "kilo-code"),
        ("opencode-kilo-provider", "kilo-code"),
        ("roo-code", "roo-code"),
        ("roocode", "roo-code"),
        ("roo code", "roo-code"),
        ("codex", "codex"),
        ("opencode", "opencode"),
        ("aider", "aider"),
        ("continue", "continue"),
    ):
        if marker in user_agent.lower():
            return agent, "infer:user-agent"
    return None, None


class SMGSticky(CustomLogger):
    async def async_pre_call_hook(self, user_api_key_dict, cache, data, call_type):
        if call_type not in SUPPORTED_CALL_TYPES:
            return data

        metadata = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
        proxy_request = data.get("proxy_server_request") or {}
        headers = normalize_headers(proxy_request.get("headers") or {})

        inferred_agent, inferred_agent_source = infer_agent(headers)

        tenant, tenant_source = first_named(
            ("metadata.tenant_id", metadata.get("tenant_id")),
            ("metadata.org_id", metadata.get("org_id")),
            ("api_key.team_id", get_attr(user_api_key_dict, "team_id")),
            ("api_key.org_id", get_attr(user_api_key_dict, "org_id")),
            ("x-tenant-id", header(headers, "x-tenant-id")),
            ("x-org-id", header(headers, "x-org-id")),
            ("default", "default"),
        )
        agent, agent_source = first_named(
            ("metadata.agent_id", metadata.get("agent_id")),
            ("x-litellm-agent-id", header(headers, "x-litellm-agent-id")),
            ("x-agent-id", header(headers, "x-agent-id")),
            ("x-assistant-id", header(headers, "x-assistant-id")),
            ("x-openwebui-agent-id", header(headers, "x-openwebui-agent-id")),
            (inferred_agent_source, inferred_agent),
            ("model", data.get("model")),
            ("unknown", "unknown"),
        )
        workspace, workspace_source = first_named(
            ("metadata.repo_id", metadata.get("repo_id")),
            ("metadata.workspace_id", metadata.get("workspace_id")),
            ("metadata.project_id", metadata.get("project_id")),
            ("x-repo-id", header(headers, "x-repo-id")),
            ("x-workspace-id", header(headers, "x-workspace-id")),
            ("x-project-id", header(headers, "x-project-id")),
            ("x-openwebui-workspace-id", header(headers, "x-openwebui-workspace-id")),
            ("x-codex-workspace-id", header(headers, "x-codex-workspace-id")),
            ("x-continue-workspace-id", header(headers, "x-continue-workspace-id")),
            ("x-roo-code-workspace-id", header(headers, "x-roo-code-workspace-id")),
            ("x-roo-workspace-id", header(headers, "x-roo-workspace-id")),
            ("x-kilo-code-workspace-id", header(headers, "x-kilo-code-workspace-id")),
            ("x-kilo-workspace-id", header(headers, "x-kilo-workspace-id")),
            ("x-kilocode-projectid", header(headers, "x-kilocode-projectid")),
            ("x-kilo-project", header(headers, "x-kilo-project")),
        )
        session, session_source = first_named(
            ("x-litellm-session-id", header(headers, "x-litellm-session-id")),
            ("x-openwebui-chat-id", header(headers, "x-openwebui-chat-id")),
            ("metadata.chat_id", metadata.get("chat_id")),
            ("metadata.session_id", metadata.get("session_id")),
            ("data.litellm_session_id", data.get("litellm_session_id")),
            ("x-codex-session-id", header(headers, "x-codex-session-id")),
            ("x-codex-thread-id", header(headers, "x-codex-thread-id")),
            ("x-codex-conversation-id", header(headers, "x-codex-conversation-id")),
            ("x-codex-window-id", header(headers, "x-codex-window-id")),
            ("x-claude-code-session-id", header(headers, "x-claude-code-session-id")),
            ("x-claude-session-id", header(headers, "x-claude-session-id")),
            ("x-continue-session-id", header(headers, "x-continue-session-id")),
            ("x-opencode-session-id", header(headers, "x-opencode-session-id")),
            ("x-opencode-session", header(headers, "x-opencode-session")),
            ("x-aider-session-id", header(headers, "x-aider-session-id")),
            ("x-roo-root-task-id", header(headers, "x-roo-root-task-id")),
            ("x-roo-code-root-task-id", header(headers, "x-roo-code-root-task-id")),
            ("x-roo-task-id", header(headers, "x-roo-task-id")),
            ("x-roo-code-task-id", header(headers, "x-roo-code-task-id")),
            ("x-roo-session-id", header(headers, "x-roo-session-id")),
            ("x-roo-code-session-id", header(headers, "x-roo-code-session-id")),
            ("x-kilo-session", header(headers, "x-kilo-session")),
            ("x-kilocode-taskid", header(headers, "x-kilocode-taskid")),
            ("x-kilocode-parent-taskid", header(headers, "x-kilocode-parent-taskid")),
            ("x-kilo-code-session-id", header(headers, "x-kilo-code-session-id")),
            ("x-kilo-session-id", header(headers, "x-kilo-session-id")),
            ("x-kilocode-session-id", header(headers, "x-kilocode-session-id")),
            ("x-kilo-code-task-id", header(headers, "x-kilo-code-task-id")),
            ("x-kilo-task-id", header(headers, "x-kilo-task-id")),
            ("x-kilocode-task-id", header(headers, "x-kilocode-task-id")),
            ("x-pi-session-id", header(headers, "x-pi-session-id")),
            ("pi:session_id", header(headers, "session_id")),
            ("x-session-affinity", header(headers, "x-session-affinity")),
            ("pi:x-client-request-id", header(headers, "x-client-request-id")),
            ("metadata.thread_id", metadata.get("thread_id")),
            ("metadata.conversation_id", metadata.get("conversation_id")),
        )
        user, user_source = first_named(
            ("metadata.user_id", metadata.get("user_id")),
            ("data.user", data.get("user")),
            ("api_key.end_user_id", get_attr(user_api_key_dict, "end_user_id")),
            ("api_key.user_id", get_attr(user_api_key_dict, "user_id")),
            ("x-user-id", header(headers, "x-user-id")),
            ("x-litellm-user-id", header(headers, "x-litellm-user-id")),
            ("x-openwebui-user-id", header(headers, "x-openwebui-user-id")),
            ("x-openwebui-user-email", header(headers, "x-openwebui-user-email")),
            ("x-codex-user-id", header(headers, "x-codex-user-id")),
            ("x-continue-user-id", header(headers, "x-continue-user-id")),
            ("x-roo-code-user-id", header(headers, "x-roo-code-user-id")),
            ("x-roo-user-id", header(headers, "x-roo-user-id")),
            ("x-kilo-code-user-id", header(headers, "x-kilo-code-user-id")),
            ("x-kilo-user-id", header(headers, "x-kilo-user-id")),
        )

        stable, stable_source = first_named(
            (session_source, session),
            (workspace_source, workspace),
            (user_source, user),
            ("data.litellm_call_id", data.get("litellm_call_id")),
        )

        raw_key = "|".join(
            [
                "v1",
                tenant or "-",
                agent or "-",
                workspace or "-",
                user or "-",
                stable_source or "-",
                stable or "-",
            ]
        )
        smg_key = "smg:" + hashlib.sha256(raw_key.encode("utf-8")).hexdigest()[:32]

        outgoing_headers = dict(data.get("headers") or {})
        for key in list(outgoing_headers):
            if str(key).lower() == SMG_ROUTING_HEADER.lower():
                outgoing_headers.pop(key)
        outgoing_headers[SMG_ROUTING_HEADER] = smg_key
        data["headers"] = outgoing_headers

        metadata["smg_sticky_key_preview"] = smg_key[:12]
        metadata["smg_sticky_sources"] = {
            "tenant": tenant_source,
            "agent": agent_source,
            "workspace": workspace_source,
            "session": session_source,
            "user": user_source,
            "stable": stable_source,
        }
        data["metadata"] = metadata
        return data


proxy_handler_instance = SMGSticky()
