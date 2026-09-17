"""A deterministic HTTP transport; tests never contact Groq."""
import json
from collections import deque

import httpx
from groq import Groq

GOOD = {
    "result": "Düzgün",
    "ideal_translation": "Я люблю свою маленькую собаку.",
    "explanation": "Məna və qrammatika düzgündür.",
    "corrections": [],
    "naturalness": "",
}


class Provider:
    def __init__(self, *items, models=None):
        self.items = deque(items)
        self.requests = []
        self.model_calls = 0
        self.models = models if models is not None else ["openai/gpt-oss-120b", "llama-3.3-70b-versatile"]
        self.client = Groq(api_key="unit-test-key", max_retries=0,
                           http_client=httpx.Client(transport=httpx.MockTransport(self.handle)))

    def handle(self, request):
        if request.url.path.endswith("/models"):
            self.model_calls += 1
            return httpx.Response(200, json={"object": "list", "data": [
                {"id": m, "object": "model", "created": 0, "owned_by": "test", "active": True}
                for m in self.models]})
        self.requests.append(json.loads(request.content))
        if not self.items:
            raise AssertionError("Unexpected API request")
        item = self.items.popleft()
        if isinstance(item, httpx.Response):
            return item
        finish = "stop"
        if isinstance(item, tuple):
            item, finish = item
        content = json.dumps(item, ensure_ascii=False) if isinstance(item, dict) else item
        return httpx.Response(200, json={
            "id": "test-completion", "object": "chat.completion", "created": 0,
            "model": self.requests[-1]["model"],
            "choices": [{"index": 0, "finish_reason": finish,
                         "message": {"role": "assistant", "content": content}}],
        })


def api_failure(status=429, code="rate_limit_exceeded"):
    return httpx.Response(status, json={"error": {
        "message": "PRIVATE_PROVIDER_BODY", "type": "invalid_request_error", "code": code}})
