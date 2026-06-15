import hashlib

from litellm.integrations.custom_logger import CustomLogger


def first(*values):
    for value in values:
        if value is not None and str(value) != "":
            return str(value)
    return None


class SMGSticky(CustomLogger):
    async def async_pre_call_hook(self, user_api_key_dict, cache, data, call_type):
        if call_type not in ("acompletion", "atext_completion", "aresponses"):
            return data

        metadata = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
        proxy_request = data.get("proxy_server_request") or {}
        incoming_headers = proxy_request.get("headers") or {}
        headers = {str(k).lower(): v for k, v in incoming_headers.items()}

        tenant = first(
            getattr(user_api_key_dict, "team_id", None),
            getattr(user_api_key_dict, "org_id", None),
            headers.get("x-tenant-id"),
            "default",
        )
        agent = first(
            metadata.get("agent_id"),
            headers.get("x-litellm-agent-id"),
            headers.get("x-assistant-id"),
            data.get("model"),
        )
        workspace = first(
            metadata.get("repo_id"),
            metadata.get("workspace_id"),
            headers.get("x-repo-id"),
            headers.get("x-workspace-id"),
        )
        session = first(
            metadata.get("session_id"),
            data.get("litellm_session_id"),
            headers.get("x-litellm-session-id"),
            headers.get("x-claude-code-session-id"),
        )
        user = first(
            data.get("user"),
            getattr(user_api_key_dict, "end_user_id", None),
            getattr(user_api_key_dict, "user_id", None),
        )

        stable = first(session, user, data.get("litellm_call_id"))
        raw_key = f"{tenant}:{agent}:{workspace}:{stable}"
        smg_key = "smg:" + hashlib.sha256(raw_key.encode("utf-8")).hexdigest()[:32]

        outgoing_headers = dict(data.get("headers") or {})
        for key in list(outgoing_headers):
            if str(key).lower() == "x-smg-routing-key":
                outgoing_headers.pop(key)
        outgoing_headers["X-SMG-Routing-Key"] = smg_key
        data["headers"] = outgoing_headers

        metadata["smg_sticky_key_preview"] = smg_key[:12]
        data["metadata"] = metadata
        return data


proxy_handler_instance = SMGSticky()
