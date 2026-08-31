"""``flightlab.cache`` -- the machinery that makes design iteration fast.

Design work is a loop: change one parameter, look at one plot, decide, change
another.  If every change reruns the vortex lattice, the loop is slow enough
that students stop exploring and start sampling four points.  So results are
cached, and the cache knows what depends on what.

The rule is simple.  Every group of parameters carries a **version**, bumped
whenever anything in it is assigned.  Every analysis mode declares which groups
it reads.  A cached result is reused when the versions of the groups it read
have not moved.  Changing the wing span bumps ``geometry``, so the
aerodynamics recompute; changing the battery mass bumps ``mass``, which the
aerodynamics never read, so they do not.

    >>> case.wing.span = 1.4
    >>> case.wing_aero()      # recomputes
    >>> case.mass["battery"] = 0.14
    >>> case.wing_aero()      # cached; mass is not an aerodynamic input

Nothing here is course content.  It exists so that
:func:`flightlab.live.explore` can move a slider at interactive speed.

Disk cache
----------
The in-memory cache lives as long as the interpreter, which is the whole
session in a notebook and one script run otherwise.  For scripts, turn on the
disk cache and a cold run reuses what a previous run computed::

    from flightlab import cache
    cache.enable_disk()          # ./.me415_cache
    cache.enable_disk("~/.cache/flightlab")

It is keyed by the *values* of the parameters read, not by version numbers, so
it survives restarts.  ``cache.clear()`` empties both.
"""

from __future__ import annotations

import hashlib
import os
import pickle
import shutil
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional, Tuple

import numpy as np

__all__ = [
    "Watched",
    "WatchedDict",
    "mode",
    "enable_disk",
    "disable_disk",
    "clear",
    "stats",
    "Stats",
]


# --- version tracking -------------------------------------------------------

_counter_lock = threading.Lock()
_global_counter = [0]


def _next_version() -> int:
    with _counter_lock:
        _global_counter[0] += 1
        return _global_counter[0]


class Watched:
    """A mutable parameter group whose version bumps on every assignment.

    Subclass it and declare the fields in ``_fields``.  Assignment to a
    declared field bumps :attr:`version`; assignment to anything else raises,
    which catches typos like ``case.wing.spam = 1.4`` that would otherwise sit
    there silently doing nothing.

    Attributes
    ----------
    version : int
        Monotonic, globally unique.  Two groups never share a version number,
        so a cache key built from several groups is unambiguous.
    """

    _fields: Tuple[str, ...] = ()

    def __init__(self, **values):
        object.__setattr__(self, "_version", _next_version())
        object.__setattr__(self, "_values", {})
        unknown = set(values) - set(self._fields)
        if unknown:
            raise AttributeError(
                f"{type(self).__name__} has no parameter(s) {sorted(unknown)}; "
                f"known parameters are {list(self._fields)}"
            )
        for name in self._fields:
            self._values[name] = values.get(name)

    # -- attribute access ---------------------------------------------------

    def __getattr__(self, name):
        values = object.__getattribute__(self, "_values")
        if name in values:
            return values[name]
        raise AttributeError(
            f"{type(self).__name__} has no parameter {name!r}; "
            f"known parameters are {list(type(self)._fields)}"
        )

    def __setattr__(self, name, value):
        if name.startswith("_"):
            object.__setattr__(self, name, value)
            return
        if name not in self._fields:
            raise AttributeError(
                f"{type(self).__name__} has no parameter {name!r}; "
                f"known parameters are {list(self._fields)}.  "
                "Assigning an unknown parameter would silently do nothing, so "
                "it is an error instead."
            )
        if _same(self._values.get(name), value):
            return  # no-op assignment does not invalidate anything
        self._values[name] = value
        object.__setattr__(self, "_version", _next_version())

    # -- introspection ------------------------------------------------------

    @property
    def version(self) -> int:
        return object.__getattribute__(self, "_version")

    def touch(self) -> None:
        """Force a version bump, e.g. after mutating a nested array in place."""
        object.__setattr__(self, "_version", _next_version())

    def asdict(self) -> Dict[str, Any]:
        """The parameter values as a plain dict."""
        return dict(self._values)

    def replace(self, **values) -> "Watched":
        """Assign several parameters at once; returns ``self`` for chaining."""
        for name, value in values.items():
            setattr(self, name, value)
        return self

    def state_key(self) -> Tuple:
        """A hashable snapshot of the values, for the disk cache."""
        return tuple(sorted((k, _hashable(v)) for k, v in self._values.items()))

    def __repr__(self) -> str:  # pragma: no cover - display only
        inner = ", ".join(
            f"{k}={_short(v)}" for k, v in self._values.items() if v is not None
        )
        return f"{type(self).__name__}({inner})"


class WatchedDict(Watched):
    """A watched group whose contents are an open-ended mapping.

    Used for the component mass table, where the keys are whatever the team's
    airplane has rather than a fixed list.

        >>> case.mass["battery"] = 0.140
        >>> case.mass["battery"]
        0.14
    """

    _fields = ()

    def __init__(self, values: Optional[Dict[str, Any]] = None):
        object.__setattr__(self, "_version", _next_version())
        object.__setattr__(self, "_values", dict(values or {}))

    def __getitem__(self, key):
        try:
            return self._values[key]
        except KeyError:
            raise KeyError(
                f"no entry {key!r}; this table has {sorted(self._values)}"
            ) from None

    def __setitem__(self, key, value):
        if key in self._values and _same(self._values[key], value):
            return
        self._values[key] = value
        object.__setattr__(self, "_version", _next_version())

    def __delitem__(self, key):
        del self._values[key]
        object.__setattr__(self, "_version", _next_version())

    def __contains__(self, key):
        return key in self._values

    def __iter__(self):
        return iter(self._values)

    def __len__(self):
        return len(self._values)

    def keys(self):
        return self._values.keys()

    def values(self):
        return self._values.values()

    def items(self):
        return self._values.items()

    def get(self, key, default=None):
        return self._values.get(key, default)

    def update(self, other) -> None:
        for key, value in dict(other).items():
            self[key] = value

    def __setattr__(self, name, value):
        if name.startswith("_"):
            object.__setattr__(self, name, value)
            return
        raise AttributeError(
            f"{type(self).__name__} is a mapping; use ['{name}'] = ... instead"
        )

    def __getattr__(self, name):
        raise AttributeError(f"{type(self).__name__} is a mapping; use [{name!r}]")

    def __repr__(self) -> str:  # pragma: no cover - display only
        return f"{type(self).__name__}({self._values!r})"


def _same(a, b) -> bool:
    """Value equality that tolerates arrays and does not raise on mismatch."""
    if a is b:
        return True
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        a_arr, b_arr = np.asarray(a), np.asarray(b)
        return a_arr.shape == b_arr.shape and bool(np.array_equal(a_arr, b_arr))
    try:
        return bool(a == b)
    except Exception:  # pragma: no cover - exotic types
        return False


def _hashable(value):
    """A hashable, value-based stand-in for the disk cache key."""
    if isinstance(value, np.ndarray):
        return ("ndarray", value.shape, value.tobytes())
    if isinstance(value, (list, tuple)):
        return tuple(_hashable(v) for v in value)
    if isinstance(value, dict):
        return tuple(sorted((k, _hashable(v)) for k, v in value.items()))
    if isinstance(value, (str, bytes, int, float, bool, type(None))):
        return value
    key = getattr(value, "state_key", None)
    if callable(key):
        return key()
    return repr(value)


def _short(value) -> str:
    if isinstance(value, np.ndarray):
        return f"array{value.shape}"
    return repr(value)


# --- the mode decorator -----------------------------------------------------


def mode(*groups: str):
    """Declare a cached analysis mode and the parameter groups it reads.

    Applied to a method of :class:`flightlab.case.Case`.  The wrapped method runs
    only when one of the named groups has changed since the last call.

    Parameters
    ----------
    *groups : str
        Names of :class:`Watched` attributes on the case.  Naming a group the
        method does not actually read is harmless but costs cache hits; failing
        to name one it *does* read gives stale results, so the modes in
        ``flightlab`` keep the list next to the code that uses it.

    Notes
    -----
    ``force=True`` on the call recomputes regardless, which is what you want
    after mutating something the version tracking cannot see -- an array
    modified in place, or a file on disk.
    """

    def decorate(func: Callable):
        name = func.__name__

        def wrapper(self, *args, force: bool = False, **kwargs):
            key = self._version_key(groups, args, kwargs)
            store = self._cache
            if not force:
                hit = store.get(name)
                if hit is not None and hit[0] == key:
                    _STATS.memory_hits += 1
                    return hit[1]
                disk = _disk_get(self, name, groups, args, kwargs)
                if disk is not None:
                    _STATS.disk_hits += 1
                    store[name] = (key, disk)
                    return disk
            _STATS.misses += 1
            result = func(self, *args, **kwargs)
            store[name] = (key, result)
            _disk_put(self, name, groups, args, kwargs, result)
            return result

        wrapper.__name__ = name
        wrapper.__doc__ = func.__doc__
        wrapper.__wrapped__ = func
        wrapper._groups = groups
        return wrapper

    return decorate


# --- statistics -------------------------------------------------------------


@dataclass
class Stats:
    """Cache hit counters, so ``stats()`` can show the loop is working."""

    memory_hits: int = 0
    disk_hits: int = 0
    misses: int = 0

    @property
    def total(self) -> int:
        return self.memory_hits + self.disk_hits + self.misses

    @property
    def hit_rate(self) -> float:
        return 0.0 if not self.total else (self.memory_hits + self.disk_hits) / self.total

    def reset(self) -> None:
        self.memory_hits = self.disk_hits = self.misses = 0

    def __repr__(self) -> str:  # pragma: no cover - display only
        return (
            f"<cache {self.memory_hits} memory + {self.disk_hits} disk hits, "
            f"{self.misses} misses ({100 * self.hit_rate:.0f}% hit rate)>"
        )


_STATS = Stats()


def stats() -> Stats:
    """The running cache counters."""
    return _STATS


# --- disk cache -------------------------------------------------------------

_disk_dir: Optional[Path] = None


def enable_disk(path: str = ".me415_cache") -> Path:
    """Turn on the on-disk result cache, so cold script runs are fast too.

    Returns the directory in use.  Entries are pickled, keyed by a hash of the
    parameter values the mode reads, and never expire -- ``clear()`` is the
    only thing that removes them.  Delete the directory freely; it is derived
    data.
    """
    global _disk_dir
    _disk_dir = Path(os.path.expanduser(path))
    _disk_dir.mkdir(parents=True, exist_ok=True)
    return _disk_dir


def disable_disk() -> None:
    """Turn the disk cache off.  Does not delete anything."""
    global _disk_dir
    _disk_dir = None


def clear(memory: bool = True, disk: bool = True) -> None:
    """Empty the caches and reset the counters."""
    if memory:
        for case in list(_LIVE_CASES):
            case._cache.clear()
    if disk and _disk_dir is not None and _disk_dir.exists():
        shutil.rmtree(_disk_dir)
        _disk_dir.mkdir(parents=True, exist_ok=True)
    _STATS.reset()


_LIVE_CASES: "set" = set()


def _register(case) -> None:
    _LIVE_CASES.add(case)


def _disk_key(case, name: str, groups: Iterable[str], args, kwargs) -> str:
    payload = (
        name,
        tuple(_hashable(getattr(case, g).state_key()) for g in groups),
        _hashable(getattr(case, "_identity", lambda: ())()),
        _hashable(args),
        _hashable(tuple(sorted(kwargs.items()))),
    )
    return hashlib.sha256(repr(payload).encode()).hexdigest()[:32]


def _disk_get(case, name, groups, args, kwargs):
    if _disk_dir is None:
        return None
    path = _disk_dir / f"{name}-{_disk_key(case, name, groups, args, kwargs)}.pkl"
    if not path.exists():
        return None
    try:
        with path.open("rb") as fh:
            return pickle.load(fh)
    except Exception:  # pragma: no cover - a corrupt entry is not fatal
        path.unlink(missing_ok=True)
        return None


def _disk_put(case, name, groups, args, kwargs, result) -> None:
    if _disk_dir is None:
        return
    path = _disk_dir / f"{name}-{_disk_key(case, name, groups, args, kwargs)}.pkl"
    try:
        with path.open("wb") as fh:
            pickle.dump(result, fh, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception:  # pragma: no cover - an unpicklable result is not fatal
        path.unlink(missing_ok=True)
