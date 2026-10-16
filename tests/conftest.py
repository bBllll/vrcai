import copy

import pytest

from vrcai.config import DEFAULTS


@pytest.fixture
def cfg():
    value = copy.deepcopy(DEFAULTS)
    value["llm"]["openai_compatible"]["api_key"] = "test-llm-secret"
    value["stt"]["groq_whisper"]["api_key"] = "test-stt-secret"
    return value
