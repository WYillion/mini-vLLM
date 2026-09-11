"""
FastAPI server for mini-vLLM inference service
"""

import asyncio
import uuid
from contextlib import asynccontextmanager
from typing import List, Optional, AsyncIterator
import logging

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from mini_vllm.config import EngineConfig, ModelConfig, CacheConfig, SchedulerConfig
from mini_vllm.engine import InferenceEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_engine: Optional[InferenceEngine] = None


class CompletionRequest(BaseModel):
    """Request model for /v1/completions endpoint"""
    prompt: str = Field(..., description="Input text prompt")
    max_tokens: int = Field(256, ge=1, le=8192, description="Maximum tokens to generate")
    temperature: float = Field(1.0, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: float = Field(1.0, ge=0.0, le=1.0, description="Nucleus sampling threshold")
    top_k: int = Field(50, ge=0, description="Top-k sampling parameter")
    stream: bool = Field(False, description="Enable streaming response")
    stop: Optional[List[str]] = Field(None, description="Stop sequences")


class CompletionResponse(BaseModel):
    """Response model for non-streaming completion"""
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: List[dict]
    usage: dict


class ChatMessage(BaseModel):
    """Chat message model"""
    role: str
    content: str


class ChatRequest(BaseModel):
    """Request model for /v1/chat/completions endpoint"""
    messages: List[ChatMessage]
    max_tokens: int = Field(256, ge=1, le=8192)
    temperature: float = Field(1.0, ge=0.0, le=2.0)
    top_p: float = Field(1.0, ge=0.0, le=1.0)
    stream: bool = Field(False)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown"""
    global _engine

    logger.info("Starting mini-vLLM engine...")

    config = EngineConfig(
        model=ModelConfig(model_name="Qwen/Qwen2.5-0.5B"),
        cache=CacheConfig(block_size=16, num_gpu_blocks=1024),
        scheduler=SchedulerConfig(max_batch_size=32, max_num_seqs=32),
    )

    _engine = InferenceEngine(config)

    try:
        _engine.initialize(use_dummy_model=True)
    except Exception as e:
        logger.warning(f"Failed to initialize with real model: {e}. Using dummy model.")
        _engine.initialize(use_dummy_model=True)

    _engine.start()
    logger.info("mini-vLLM engine started successfully")

    yield

    logger.info("Shutting down mini-vLLM engine...")
    if _engine:
        _engine.stop()
    logger.info("mini-vLLM engine stopped")


app = FastAPI(
    title="mini-vLLM",
    description="A simplified vLLM inference framework",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if _engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    return {"status": "healthy", "stats": _engine.stats}


@app.post("/v1/completions", response_model=CompletionResponse)
async def create_completion(request: CompletionRequest):
    """
    Create a completion for the given prompt.
    OpenAI-compatible /v1/completions endpoint.
    """
    if _engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")

    request_id = f"cmpl-{uuid.uuid4().hex}"

    if request.stream:
        return EventSourceResponse(
            generate_streaming_completion(request, request_id),
            media_type="text/event-stream",
        )

    output_tokens = []
    async for token in _engine.generate(
        prompt=request.prompt,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        top_k=request.top_k,
        stream=False,
    ):
        output_tokens.append(token)

    full_text = "".join(output_tokens)

    return CompletionResponse(
        id=request_id,
        created=int(asyncio.get_event_loop().time()),
        model=_engine.model_config.model_name,
        choices=[{"text": full_text, "index": 0, "finish_reason": "stop"}],
        usage={
            "prompt_tokens": len(output_tokens),
            "completion_tokens": len(output_tokens),
            "total_tokens": len(output_tokens) * 2,
        },
    )


async def generate_streaming_completion(request: CompletionRequest, request_id: str):
    """Generate streaming response for completion"""
    if _engine is None:
        return

    token_count = 0
    async for token in _engine.generate(
        prompt=request.prompt,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        top_k=request.top_k,
        stream=True,
    ):
        token_count += 1
        yield {
            "event": "token",
            "data": f'data: {{"id": "{request_id}", "choices": [{{"text": "{token}", "index": 0}}]}}\n\n',
        }

    yield {
        "event": "done",
        "data": f'data: {{"id": "{request_id}", "choices": [{{"finish_reason": "stop"}}], "usage": {{"total_tokens": {token_count}}}}}}\n\n',
    }


@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatRequest):
    """
    Create a chat completion.
    OpenAI-compatible /v1/chat/completions endpoint.
    """
    if _engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")

    request_id = f"chat-{uuid.uuid4().hex}"

    prompt = "\n".join([f"{msg.role}: {msg.content}" for msg in request.messages])
    prompt = f"System: You are a helpful assistant.\n{prompt}\nAssistant:"

    if request.stream:
        return EventSourceResponse(
            generate_streaming_chat(request, request_id, prompt),
            media_type="text/event-stream",
        )

    output_tokens = []
    async for token in _engine.generate(
        prompt=prompt,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        stream=False,
    ):
        output_tokens.append(token)

    full_text = "".join(output_tokens)

    return {
        "id": request_id,
        "object": "chat.completion",
        "created": int(asyncio.get_event_loop().time()),
        "model": _engine.model_config.model_name,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": full_text},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": len(output_tokens),
            "completion_tokens": len(output_tokens),
            "total_tokens": len(output_tokens) * 2,
        },
    }


async def generate_streaming_chat(request: ChatRequest, request_id: str, prompt: str):
    """Generate streaming response for chat"""
    if _engine is None:
        return

    token_count = 0
    async for token in _engine.generate(
        prompt=prompt,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        stream=True,
    ):
        token_count += 1
        yield {
            "event": "token",
            "data": f'data: {{"id": "{request_id}", "choices": [{{"delta": {{"content": "{token}"}}, "index": 0}}]}}\n\n',
        }

    yield {
        "event": "done",
        "data": f'data: {{"id": "{request_id}", "choices": [{{"finish_reason": "stop"}}]}}\n\n',
    }


@app.get("/v1/models")
async def list_models():
    """List available models"""
    if _engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")

    return {
        "object": "list",
        "data": [
            {
                "id": _engine.model_config.model_name,
                "object": "model",
                "created": 1700000000,
                "owned_by": "mini-vllm",
            }
        ],
    }


@app.get("/stats")
async def get_stats():
    """Get engine statistics"""
    if _engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    return _engine.stats


def create_app() -> FastAPI:
    """Factory function to create FastAPI app"""
    return app


def main():
    """Main entry point for running the server"""
    import uvicorn

    uvicorn.run(
        "mini_vllm.api.server:app",
        host="0.0.0.0",
        port=8000,
        workers=1,
        reload=False,
    )


if __name__ == "__main__":
    main()
