import abc
import typing as typ

import datasets
from datasets import fingerprint

from loguru import logger

from throughster.base import ModelInterface


class HasNoFingerprint(typ.Protocol):
    """Protocol for objects that have a `_no_fingerprint` attribute."""

    _no_fingerprint: list[str]


def hexdigest(obj: object | HasNoFingerprint) -> str:
    """Computes a fingerprint for the operation.

    NOTE: This allows computing deterministic hashes for `datasets.map()` and `datasets.filter()` operations.
        This is useful for caching transformations properly.
    """
    hasher_ = fingerprint.Hasher()

    # Hash base python types
    if isinstance(obj, list):
        for v_ in obj:
            hasher_.update(v_)
        return hasher_.hexdigest()
    if isinstance(obj, dict):
        for k_, v_ in sorted(obj.items()):
            hasher_.update(k_)
            hasher_.update(hexdigest(v_))
        return hasher_.hexdigest()
    if isinstance(obj, set):
        for v_ in sorted(obj):
            hasher_.update(hexdigest(v_))
        return hasher_.hexdigest()
    if not hasattr(obj, "__getstate__"):
        hasher_.update(obj)
        return hasher_.hexdigest()

    # Get the state and rely on defaults hasing strategies if none is found
    state: dict | None = obj.__getstate__()  # type: ignore
    if state is None:
        hasher_.update(obj)
        return hasher_.hexdigest()

    # Hash the class
    hasher_.update(obj.__class__)

    # Hash the state
    _no_fingerprint = getattr(obj, "_no_fingerprint", [])
    for k in sorted(state):
        if k in _no_fingerprint:
            continue
        hasher_.update(k)
        hasher_.update(hexdigest(state[k]))
    return hasher_.hexdigest()


D = typ.TypeVar("D", bound=typ.Union[datasets.Dataset, datasets.DatasetDict])


class HfOperation(abc.ABC):
    """Base class for operations compatible with `datasets.Dataset.map(..., batched=True)`."""

    _no_fingerprint: None | list[str] = []
    _client = None

    def __init__(self, init_client_fn: typ.Callable[..., ModelInterface], **kwargs: typ.Any):
        self.init_client_fn = init_client_fn

    @property
    def client(self) -> ModelInterface:
        """NOTE: Lazy initialization of ModelInterface.
        When running multi-threaded operations with hf .map(), we need to initialize a ModelInterface in each thread.
        """
        if self._client:
            return self._client
        return self.init_client_fn()

    @abc.abstractmethod
    def __call__(self, batch: dict[str, list[typ.Any]], idx: None | list[int] = None) -> dict[str, list[typ.Any]]:
        """Transforms an input batch into a new batch.

        NOTE: The resulting batch potentially contains fewer and/or new **keys**.
        NOTE: The resulting batch potentially contains fewer and/or new **rows**.
        """
        ...


def transform(
    data: D,
    fn: type[HfOperation] | typ.Callable,
    operation: typ.Literal["map", "filter"],
    **kws: typ.Any,
) -> D:
    """Applies an operation to a dataset and computes a fingerprint.

    NOTE: replaces `datasets.fingerprint.hashregister` -- it's depreciated.
    """
    if isinstance(data, datasets.DatasetDict):
        outputs = {}
        for split, d in data.items():  # type: ignore
            kws_ = kws.copy()
            if "desc" in kws_:
                kws_["desc"] = f"[{split}] {kws['desc']}"
            outputs[split] = transform(d, fn, operation=operation, **kws_)
        return datasets.DatasetDict(outputs)  # type: ignore

    # Compute the fingerpint
    # Most kws are left out, this differs from HF's implementation which uses all of them
    new_fingerprint = hexdigest(
        [
            operation,
            data._fingerprint,
            hexdigest(fn),
            {k: v for k, v in kws.items() if k in {"fn_kwargs"}},
        ]
    )

    # apply the function
    fn_name = fn.__name__ if hasattr(fn, "__name__") else fn.__class__.__name__
    logger.debug(
        "Applying {fn} with new_fingerprint={new_fingerprint}",
        fn=f"dataset.{operation}({fn_name}, ...)",
        new_fingerprint=new_fingerprint,
    )
    dset_fn = {"map": data.map, "filter": data.filter}[operation]
    return dset_fn(fn, new_fingerprint=new_fingerprint, **kws)
