import ast
import hashlib
import inspect
import json
import textwrap


def _annotation_text(annotation: ast.expr | None) -> str | None:
    if annotation is None:
        return None
    return ast.unparse(annotation)


def _parameter(
    argument: ast.arg,
    kind: str,
    default: ast.expr | None = None,
    *,
    required: bool = True,
) -> dict[str, object]:
    parameter: dict[str, object] = {
        "annotation": _annotation_text(argument.annotation),
        "kind": kind,
        "name": argument.arg,
    }
    if not required:
        if default is None:
            raise ValueError("Optional signature parameter is missing its default")
        parameter["default"] = ast.unparse(default)
    return parameter


def _source_signature(method: object) -> dict[str, object]:
    source = textwrap.dedent(inspect.getsource(inspect.unwrap(method)))
    module = ast.parse(source)
    function = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
    )
    arguments = function.args
    positional = [*arguments.posonlyargs, *arguments.args]
    default_offset = len(positional) - len(arguments.defaults)
    parameters = []
    for index, argument in enumerate(positional):
        kind = (
            "positional_only"
            if index < len(arguments.posonlyargs)
            else "positional_or_keyword"
        )
        has_default = index >= default_offset
        default = arguments.defaults[index - default_offset] if has_default else None
        parameters.append(
            _parameter(
                argument,
                kind,
                default,
                required=not has_default,
            )
        )
    if arguments.vararg is not None:
        parameters.append(_parameter(arguments.vararg, "var_positional"))
    for argument, default in zip(
        arguments.kwonlyargs,
        arguments.kw_defaults,
        strict=True,
    ):
        parameters.append(
            _parameter(
                argument,
                "keyword_only",
                default,
                required=default is None,
            )
        )
    if arguments.kwarg is not None:
        parameters.append(_parameter(arguments.kwarg, "var_keyword"))
    return {
        "async": isinstance(function, ast.AsyncFunctionDef),
        "parameters": parameters,
        "return": _annotation_text(function.returns),
    }


def signature_digest(methods: dict[str, object]) -> str:
    payload = {
        name: _source_signature(method) for name, method in sorted(methods.items())
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()
