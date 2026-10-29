"""Small shared helpers for complete callable screening without classifier imports."""

from __future__ import annotations

import functools
import inspect
import math
from numbers import Real
from typing import Any


def validate_threshold(value) -> None:
    """Reject configuration that could silently disable blocking."""
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(value)
        or not 0 <= float(value) <= 1
    ):
        raise ValueError("conf_threshold must be finite and between 0.0 and 1.0")


def argument_builder(func):
    """Inspect once, then bind every call including defaults and variadic inputs."""
    prefix: tuple[Any, ...] = ()
    fixed_kwargs: dict[str, Any] = {}
    if isinstance(func, functools.partial):
        prefix, fixed_kwargs, func = func.args, func.keywords or {}, func.func
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        signature = None
    placeholder = getattr(functools, "Placeholder", None)

    def arguments(args, kwargs):
        remaining = iter(args)
        resolved: list[Any] = []
        for value in prefix:
            if placeholder is not None and value is placeholder:
                try:
                    value = next(remaining)
                except StopIteration as exc:
                    raise TypeError(
                        "Missing positional argument for partial placeholder"
                    ) from exc
            resolved.append(value)
        args = (*resolved, *remaining)
        kwargs = {**fixed_kwargs, **kwargs}
        if signature is None:
            return {"args": list(args), "kwargs": kwargs} if args else dict(kwargs)
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        payload = {}
        extras = {}
        for name, value in bound.arguments.items():
            parameter = signature.parameters[name]
            if parameter.kind == inspect.Parameter.VAR_KEYWORD:
                extras = value
            elif parameter.kind == inspect.Parameter.VAR_POSITIONAL:
                if value:
                    payload[name] = list(value)
            else:
                if name in {"self", "cls"} and name == next(iter(signature.parameters)):
                    owner = value if isinstance(value, type) else type(value)
                    member = inspect.getattr_static(
                        owner, getattr(func, "__name__", ""), None
                    )
                    if isinstance(member, classmethod):
                        member = member.__func__
                    if (
                        not isinstance(member, staticmethod)
                        and callable(member)
                        and inspect.unwrap(member) is inspect.unwrap(func)
                    ):
                        continue
                payload[name] = value
        if payload.keys() & extras.keys():
            # A **kwargs key must never overwrite screened positional content.
            return {"args": list(args), "kwargs": dict(kwargs)}
        payload.update(extras)
        return payload

    return arguments
