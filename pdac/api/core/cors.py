from fastapi.middleware.cors import CORSMiddleware

# Production frontend + local dev
ALLOWED_ORIGINS = [
    "https://www.panopticai.online",
    "https://panopticai.online",
    "http://localhost:8080",
    "http://localhost:5173",
    "http://127.0.0.1:8080",
    "http://127.0.0.1:5173",
]

def add_cors(app):
    app.add_middleware(
        CORSMiddleware,
        allow_origins=ALLOWED_ORIGINS,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["*"],
        max_age=86400,
    )

