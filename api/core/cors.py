# api/core/cors.py
from __future__ import annotations
import os
from fastapi.middleware.cors import CORSMiddleware

def add_cors(app):
    origins = os.getenv("CORS_ORIGINS", "*")
    allow_origins = [o.strip() for o in origins.split(",")] if origins else ["*"]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allow_origins if allow_origins != ["*"] else ["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
