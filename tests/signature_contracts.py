import hashlib
import inspect
import json


def signature_digest(methods: dict[str, object]) -> str:
    payload = {
        name: str(inspect.signature(method)) for name, method in sorted(methods.items())
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()
