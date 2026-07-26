def test_app_importable():
    """Dependency-install sanity check: app.main must expose a FastAPI instance."""
    from fastapi import FastAPI

    from app.main import app

    assert isinstance(app, FastAPI)
