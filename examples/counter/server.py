import argparse
import aiocache
import fastapi

app = fastapi.FastAPI()
cache = aiocache.Cache()
KEYS_ = "__keys__"


async def _get_keys(cache: aiocache.Cache | aiocache.BaseCache) -> list[str]:
    """Get the keys."""
    keys = await cache.get(KEYS_)  # type: ignore
    if keys is None:
        return []
    if not isinstance(keys, str):
        raise TypeError(f"Expected `keys` to be a string, got `{type(keys)}`")
    return keys.split(",")


@app.get("/")
def get_root() -> dict[str, str]:
    """Return the root."""
    return {"status": "ok"}


@app.get("/get")
async def get_summary() -> dict[str, int]:
    """Return the summary."""
    keys = await _get_keys(cache)
    output = {}
    for key in keys:
        output[key] = await cache.get(key)
    return output


@app.put("/increment")
async def increment(data: dict[str, int]) -> None:
    """Increment the counter."""
    for k, v in data.items():
        keys = await _get_keys(cache)
        if k not in keys:
            keys.append(k)
            await cache.set(KEYS_, ",".join(keys))
            await cache.set(k, 0)
        await cache.increment(k, int(v))


@app.delete("/reset")
async def reset_counters() -> None:
    """Reset the counters."""
    keys = await _get_keys(cache)
    for key in keys:
        await cache.set(key, 0)
    await cache.set(KEYS_, "")


@app.get("/health")
def health() -> dict[str, str]:
    """Return the health status of the server."""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn

    args = argparse.ArgumentParser()
    args.add_argument("--port", type=int, default=8000)
    args = args.parse_args()

    uvicorn.run(app, port=args.port)
