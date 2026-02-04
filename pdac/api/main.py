import logging
from fastapi import FastAPI
from pdac.api.core.cors import add_cors
from pdac.api.routers.health import router as health_router
from pdac.api.routers.predict import router as predict_router

logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s %(message)s")

app = FastAPI(title="PANOPTIC PDAC API", version="1.0.0")
add_cors(app)

app.include_router(health_router)
app.include_router(predict_router)

