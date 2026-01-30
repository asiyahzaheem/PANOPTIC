from fastapi import FastAPI
from api.core.cors import add_cors
from api.routers.health import router as health_router
from api.routers.predict import router as predict_router

app = FastAPI(title="PANOPTIC PDAC API", version="1.0.0")
add_cors(app)

app.include_router(health_router)
app.include_router(predict_router)
