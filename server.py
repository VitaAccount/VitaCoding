from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional
import os

app = FastAPI()

class ChatRequest(BaseModel):
    message: str
    temperature: Optional[float] = 0.7
    max_length: Optional[int] = 512
    top_p: Optional[float] = 0.9
    top_k: Optional[int] = 50

class ChatResponse(BaseModel):
    response: str

# Load model and tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Try to load the continuously learned model, fall back to pre-trained if not available
MODEL_PATH = "./continuous_learner"
if os.path.exists(MODEL_PATH):
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
else:
    model = AutoModelForCausalLM.from_pretrained("facebook/opt-350m")
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")

model.to(device)
model.eval()

def generate_response(
    message: str,
    temperature: float = 0.7,
    max_length: int = 512,
    top_p: float = 0.9,
    top_k: int = 50,
) -> str:
    inputs = tokenizer(
        message,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    ).to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_return_sequences=1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    try:
        response = generate_response(
            request.message,
            temperature=request.temperature,
            max_length=request.max_length,
            top_p=request.top_p,
            top_k=request.top_k,
        )
        return ChatResponse(response=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 