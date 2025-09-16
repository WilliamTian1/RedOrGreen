from pydantic import BaseModel


class AnalyzeRequest(BaseModel):
    text: str


class AnalyzeResponse(BaseModel):
    sector: str
    sentiment: str
    sentiment_confidence: float
    prediction: str
    prediction_confidence: float




