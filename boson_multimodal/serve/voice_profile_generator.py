"""Voice-consistent streaming generator for TTS pipeline integration."""

import asyncio
import torch
from typing import AsyncIterator, Optional
from queue import Queue, Empty

from .serve_engine import HiggsAudioServeEngine, HiggsAudioStreamerDelta
from .voice_profile_session import VoiceProfileSession


class VoiceProfileGenerator:
    """
    Drop-in generator for voice-consistent TTS that reads from a sentence queue.

    Usage:
        generator = VoiceProfileGenerator(
            engine=engine,
            speaker_desc="Female, British accent, calm tone, clear audio",
            scene_prompt="Quiet studio recording",
        )

        # Feed sentences from your pipeline
        generator.sentence_queue.put("Hello, welcome!")
        generator.sentence_queue.put("How are you today?")
        generator.sentence_queue.put(None)  # Signal end

        # Consume audio tokens
        async for delta in generator.stream():
            if delta.audio_tokens is not None:
                # Process audio tokens in your pipeline
                yield delta.audio_tokens
    """

    def __init__(
        self,
        engine: HiggsAudioServeEngine,
        speaker_desc: str,
        scene_prompt: Optional[str] = None,
        generation_chunk_buffer_size: int = 2,
        max_new_tokens: int = 2048,
        temperature: float = 1.0,
        top_p: float = 0.95,
        top_k: Optional[int] = 50,
    ):
        self.engine = engine
        self.session = engine.create_voice_profile_session(
            speaker_desc=speaker_desc,
            scene_prompt=scene_prompt,
            generation_chunk_buffer_size=generation_chunk_buffer_size,
        )
        self.sentence_queue: Queue = Queue()

        # Generation params
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k

    async def stream(self) -> AsyncIterator[HiggsAudioStreamerDelta]:
        """
        Stream audio generation from queued sentences.

        Yields HiggsAudioStreamerDelta for each token generated.
        Put None in sentence_queue to signal completion.
        """
        while True:
            try:
                text = self.sentence_queue.get(timeout=0.1)
            except Empty:
                await asyncio.sleep(0.01)
                continue

            if text is None:  # End signal
                break

            async for delta in self.engine.generate_delta_stream_with_voice_profile(
                text=text,
                session=self.session,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
            ):
                yield delta

    def reset_session(self):
        """Reset voice context while keeping the same voice profile."""
        self.session.reset()

    @property
    def generation_count(self) -> int:
        """Number of generations completed in this session."""
        return self.session.generation_count
