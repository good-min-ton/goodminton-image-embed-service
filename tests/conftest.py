import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="session")
def client():
    """One TestClient for the whole test session.

    Entering it as a context manager runs the app's lifespan, which loads
    the real SigLIP model once (a few seconds) instead of once per test.
    """
    with TestClient(app) as c:
        yield c
