import asyncio
import sys
import numpy as np
from boson_multimodal.serve.serve_engine import HiggsAudioServeEngine
from boson_multimodal.data_types import ChatMLSample, Message
from boson_multimodal.model.higgs_audio.utils import revert_delay_pattern

import torch

MODEL_PATH = "bosonai/higgs-audio-v2-generation-3B-base"
AUDIO_TOKENIZER_PATH = "bosonai/higgs-audio-v2-tokenizer"

system_prompt = (
    "Generate audio following instruction.\n\n<|scene_desc_start|>\nAudio is recorded from a quiet room.\n<|scene_desc_end|>"
)

messages = [
    Message(
        role="system",
        content=system_prompt,
    ),
    Message(
        role="user",
        content="The sun rises in the east and sets in the west. This simple fact has been observed by humans for thousands of years.",
    ),
]
device = "cuda" if torch.cuda.is_available() else "cpu"

serve_engine = HiggsAudioServeEngine(MODEL_PATH, AUDIO_TOKENIZER_PATH, device=device)

# Global buffer to accumulate audio tokens for streaming
audio_token_buffer = []
chunk_size = 64

async def process_audio_tokens_to_pcm(delta, serve_engine):
    global audio_token_buffer

    # print(f"delta: {delta}", file=sys.stderr)

    if delta.audio_tokens is not None:
        audio_token_buffer.append(delta.audio_tokens)
    
    if len(audio_token_buffer) >= chunk_size or delta.text == "<|eot_id|>":
        audio_chunk = torch.stack(audio_token_buffer[:chunk_size], dim=1)
        num_codebooks = audio_chunk.shape[0]
        
        vq_code = revert_delay_pattern(audio_chunk).clip(0, serve_engine.audio_codebook_size - 1)
        wv_numpy = serve_engine.audio_tokenizer.decode(vq_code.unsqueeze(0))[0, 0]
        pcm_data = (wv_numpy * 32767).astype(np.int16)
        
        # Write to stdout asynchronously
        loop = asyncio.get_running_loop()
        loop.run_in_executor(None, sys.stdout.buffer.write, pcm_data.tobytes())
        loop.run_in_executor(None, sys.stdout.buffer.flush)
    
        # Calculate how many tokens to keep for next chunk
        # We need to preserve the tokens that were cut off by the delay pattern
        # Keep the last (num_codebooks - 1) tokens to maintain continuity
        tokens_to_keep = num_codebooks - 1
        audio_token_buffer = audio_token_buffer[chunk_size - tokens_to_keep:]

async def main():
    streamer = serve_engine.generate_delta_stream(
        chat_ml_sample=ChatMLSample(messages=messages),
        temperature=0.75,
        top_p=0.95,
        top_k=50,
        stop_strings=["<|end_of_text|>", "<|eot_id|>"],
        ras_win_len=7,
        ras_win_max_num_repeat=2,
        force_audio_gen=True,
    )

    async for delta in streamer:
        await process_audio_tokens_to_pcm(delta, serve_engine)

if __name__ == "__main__":
    asyncio.run(main())