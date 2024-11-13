from pydantic import BaseModel

HELLO_WORLD_PROMPT = "write hello world in python"
HELLO_WORLD_MESSAGES = [{"role": "user", "content": HELLO_WORLD_PROMPT}]
HELLO_WORLD_MAX_TOKENS = 1000


class TestGuidedGeneration(BaseModel):
    __test__ = False
    hello_world: str


class TestGuidedGenerationFail(BaseModel):
    __test__ = False
    helloworld: str
