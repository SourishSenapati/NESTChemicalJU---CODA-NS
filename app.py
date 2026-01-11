from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import neuro_engine # Imports your engine

app = FastAPI()

# Allow frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class InputData(BaseModel):
    site_id: int
    missing: int
    queries: int
    sae: int
    protocol_text: str = "" # <--- Added this (Gap A)

@app.post("/analyze_site")
def run_analysis(data: InputData):
    print(f"Received request for Site {data.site_id}")
    result = neuro_engine.analyze(data.site_id, data.missing, data.queries, data.sae, data.protocol_text)
    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
