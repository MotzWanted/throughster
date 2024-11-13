import pathlib
import subprocess
import typing as typ
import time
import requests

script = (pathlib.Path(__file__).parent / "server.py").absolute()


class CounterClient:
    """A counter server."""

    host = "http://localhost"

    def __init__(self, port: int = 8123):
        self.port = port

    def increment(self, data: dict[str, float]) -> None:
        """Increment the counter."""
        requests.put(f"{self.host}:{self.port}/increment", json=data, timeout=3.0)

    def get(self) -> dict[str, float]:
        """Return the counter."""
        response = requests.get(f"{self.host}:{self.port}/get", timeout=3.0)
        response.raise_for_status()
        return response.json()

    def reset(self) -> None:
        """Reset the counter."""
        response = requests.delete(f"{self.host}:{self.port}/reset", timeout=3.0)
        response.raise_for_status()

    def is_healthy(self) -> bool:
        """Check if the server is up and healthy."""
        try:
            response = requests.get(f"{self.host}:{self.port}/health", timeout=3.0)
            response.raise_for_status()
            return response.json().get("status") == "healthy"
        except requests.RequestException:
            return False


class CounterServer(CounterClient):
    """A counter server."""

    def __init__(self, port: int = 8123):
        super().__init__(port)
        self.proc: None | subprocess.Popen = None

    def __enter__(self) -> typ.Self:
        self.proc = subprocess.Popen(
            [
                "python",
                str(script),
                "--port",
                str(self.port),
            ],
            stdout=open("server_stdout.log", "w"),
            stderr=open("server_stderr.log", "w"),
        )

        # Ensure the server is up and running with a health check
        for _ in range(60):  # Wait for up to 6 seconds
            if self.is_healthy():
                break
            time.sleep(0.1)
        else:
            self.__exit__()  # Clean up the process if startup fails
            raise RuntimeError("Server failed to start.")

        return self

    def __exit__(self, *args) -> None:
        self.reset()
        if self.proc is not None:
            self.proc.terminate()
            self.proc.wait()
            self.proc = None

    def get_client(self) -> CounterClient:
        """Return a client."""
        return CounterClient(self.port)
