# Plan: Voice-Consistent Streaming for HiggsAudioServeEngine

## Overview

This document outlines the implementation plan for adding a **voice-consistent streaming method** to `HiggsAudioServeEngine` that uses the **text profile/description method** (speaker_desc + scene_prompt) while maintaining voice consistency across multiple generations.

---

## Problem Statement

Currently, Higgs Audio supports two methods for voice generation:

1. **Voice Clone**: Uses a `.wav` file + text transcript to clone a voice. This naturally maintains consistency because the cloned voice is deterministic.

2. **Text Profile/Description**: Uses `speaker_desc` (voice description) and `scene_prompt` (environment description) to generate a voice. **This method generates a slightly different voice for each generation**, which breaks consistency across multiple audio generations.

The goal is to enable the text profile method to maintain **consistent voice characteristics across streaming generations**, similar to how voice cloning works.

---

## Solution Approach

Based on analysis of `boson_multimodal/examples/generation.py` (specifically the `HiggsAudioModelClient` class), voice consistency is achieved by:

1. **Accumulating generated audio tokens** from previous generations as context
2. **Accumulating conversation messages** (user prompts + assistant audio responses)
3. **Using a buffer size** to limit context growth (`generation_chunk_buffer_size`)
4. **Passing accumulated context** to subsequent generation calls

### Key Code Pattern from `generation.py`

```python
# From HiggsAudioModelClient.generate() - lines 268-370

generated_audio_ids = []  # Stores audio tokens from all generations
generation_messages = []  # Stores user/assistant message pairs

for idx, chunk_text in enumerate(chunked_text):
    # 1. Add user message
    generation_messages.append(Message(role="user", content=chunk_text))

    # 2. Build sample with base messages + accumulated generation messages
    chatml_sample = ChatMLSample(messages=messages + generation_messages)

    # 3. Combine reference audio + all generated audio as context
    context_audio_ids = audio_ids + generated_audio_ids

    # 4. Prepare input with accumulated context
    curr_sample = ChatMLDatasetSample(
        input_ids=torch.LongTensor(input_tokens),
        audio_ids_concat=torch.concat([ele.cpu() for ele in context_audio_ids], dim=1),
        audio_ids_start=torch.cumsum(...),
        ...
    )

    # 5. Generate audio
    outputs = self._model.generate(**batch, ...)

    # 6. Process and store generated audio tokens
    audio_out_ids = revert_delay_pattern(outputs[1][0])
    audio_out_ids = audio_out_ids.clip(0, codebook_size - 1)[:, 1:-1]
    generated_audio_ids.append(audio_out_ids)

    # 7. Add assistant response placeholder
    generation_messages.append(Message(role="assistant", content=AudioContent(audio_url="")))

    # 8. Buffer management - keep only recent context
    if generation_chunk_buffer_size and len(generated_audio_ids) > generation_chunk_buffer_size:
        generated_audio_ids = generated_audio_ids[-generation_chunk_buffer_size:]
        generation_messages = generation_messages[(-2 * generation_chunk_buffer_size):]
```

---

## Implementation Plan

### Part 1: Create Voice Profile Session Manager

Create a new class `VoiceProfileSession` to manage state across streaming generations.

**File**: `boson_multimodal/serve/voice_profile_session.py` (new file)

```python
from dataclasses import dataclass, field
from typing import List, Optional
import torch

from ..data_types import Message, AudioContent, ChatMLSample

@dataclass
class VoiceProfileConfig:
    """Configuration for a voice profile session."""
    speaker_desc: str  # e.g., "Male, American accent, moderate pitch, friendly tone"
    scene_prompt: Optional[str] = None  # e.g., "Audio is recorded from a quiet room."
    generation_chunk_buffer_size: Optional[int] = 3  # Number of past generations to keep as context


@dataclass
class VoiceProfileSession:
    """Manages state for voice-consistent generation across multiple streaming calls."""

    config: VoiceProfileConfig

    # Accumulated state
    generated_audio_ids: List[torch.Tensor] = field(default_factory=list)
    generation_messages: List[Message] = field(default_factory=list)

    # Base messages (system prompt with voice profile)
    base_messages: List[Message] = field(default_factory=list)

    # Session metadata
    generation_count: int = 0

    def __post_init__(self):
        """Initialize base messages with voice profile."""
        self._build_base_messages()

    def _build_base_messages(self):
        """Build the system message with voice profile."""
        scene_desc_parts = []

        # Add scene prompt if provided
        if self.config.scene_prompt:
            scene_desc_parts.append(self.config.scene_prompt)

        # Add speaker description
        scene_desc_parts.append(f"SPEAKER0: {self.config.speaker_desc}")

        scene_desc = "\n\n".join(scene_desc_parts)

        system_content = (
            "Generate audio following instruction.\n\n"
            f"<|scene_desc_start|>\n{scene_desc}\n<|scene_desc_end|>"
        )

        self.base_messages = [Message(role="system", content=system_content)]

    def get_context_messages(self) -> List[Message]:
        """Get all messages including base and accumulated generation messages."""
        return self.base_messages + self.generation_messages

    def get_context_audio_ids(self) -> List[torch.Tensor]:
        """Get accumulated audio IDs for context."""
        return self.generated_audio_ids.copy()

    def add_user_message(self, text: str):
        """Add a user message to the generation context."""
        self.generation_messages.append(Message(role="user", content=text))

    def add_generated_audio(self, audio_ids: torch.Tensor):
        """Add generated audio tokens and assistant message to context."""
        self.generated_audio_ids.append(audio_ids)
        self.generation_messages.append(
            Message(role="assistant", content=AudioContent(audio_url=""))
        )
        self.generation_count += 1

        # Apply buffer management
        self._apply_buffer_limit()

    def _apply_buffer_limit(self):
        """Trim context to buffer size limit."""
        buffer_size = self.config.generation_chunk_buffer_size
        if buffer_size is not None and len(self.generated_audio_ids) > buffer_size:
            self.generated_audio_ids = self.generated_audio_ids[-buffer_size:]
            # Each generation adds 2 messages (user + assistant)
            self.generation_messages = self.generation_messages[(-2 * buffer_size):]

    def reset(self):
        """Reset the session state, keeping the voice profile configuration."""
        self.generated_audio_ids = []
        self.generation_messages = []
        self.generation_count = 0
```

---

### Part 2: Add Voice-Consistent Streaming Method to HiggsAudioServeEngine

Add a new method `generate_delta_stream_with_voice_profile` to `serve_engine.py`.

**Key additions to `serve_engine.py`**:

```python
# New imports at top of file
from .voice_profile_session import VoiceProfileSession, VoiceProfileConfig

# New dataclass for streaming response with accumulated tokens
@dataclass
class HiggsAudioStreamerDeltaWithContext(HiggsAudioStreamerDelta):
    """Extended delta that tracks accumulated audio tokens for voice consistency."""
    accumulated_audio_tokens: Optional[torch.Tensor] = None
    generation_index: int = 0


class HiggsAudioServeEngine:
    # ... existing code ...

    def _prepare_inputs_with_context(
        self,
        chat_ml_sample: ChatMLSample,
        context_audio_ids: List[torch.Tensor],
        force_audio_gen: bool = False
    ):
        """
        Prepare inputs with accumulated audio context for voice consistency.

        This is similar to _prepare_inputs but handles multiple audio contexts
        from previous generations.
        """
        input_tokens, _, audio_contents, _ = prepare_chatml_sample(
            chat_ml_sample,
            self.tokenizer,
        )

        postfix = "<|start_header_id|>assistant<|end_header_id|>\n\n"
        if force_audio_gen:
            postfix += "<|audio_out_bos|>"
        postfix = self.tokenizer.encode(postfix, add_special_tokens=False)
        input_tokens.extend(postfix)

        # Process audio contents from chat (voice clone references if any)
        audio_ids_l = []
        for audio_content in audio_contents:
            if audio_content.audio_url not in ["placeholder", ""]:
                raw_audio, _ = librosa.load(audio_content.audio_url, sr=self.audio_tokenizer.sampling_rate)
            elif audio_content.raw_audio is not None:
                raw_audio, _ = librosa.load(
                    BytesIO(base64.b64decode(audio_content.raw_audio)),
                    sr=self.audio_tokenizer.sampling_rate
                )
            else:
                raw_audio = None

            if raw_audio is not None:
                audio_ids = self.audio_tokenizer.encode(raw_audio, self.audio_tokenizer.sampling_rate)
                audio_ids_l.append(audio_ids.squeeze(0).cpu())

        # Add accumulated context audio IDs
        for ctx_audio_ids in context_audio_ids:
            audio_ids_l.append(ctx_audio_ids.cpu() if ctx_audio_ids.is_cuda else ctx_audio_ids)

        if len(audio_ids_l) > 0:
            audio_ids_start = torch.tensor(
                np.cumsum(np.array([0] + [audio_ids.shape[1] for audio_ids in audio_ids_l])),
                dtype=torch.long,
                device=self.device,
            )[0:-1]
            audio_ids_concat = torch.cat(audio_ids_l, dim=1)
        else:
            audio_ids_start = None
            audio_ids_concat = None

        sample = ChatMLDatasetSample(
            input_ids=torch.LongTensor(input_tokens),
            label_ids=None,
            audio_ids_concat=audio_ids_concat,
            audio_ids_start=audio_ids_start,
            audio_waveforms_concat=None,
            audio_waveforms_start=None,
            audio_sample_rate=None,
            audio_speaker_indices=None,
        )
        data = self.collator([sample])
        inputs = asdict(data)
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(self.model.device)

        return inputs

    async def generate_delta_stream_with_voice_profile(
        self,
        text: str,
        session: VoiceProfileSession,
        max_new_tokens: int = 2048,
        temperature: float = 0.7,
        top_k: Optional[int] = None,
        top_p: float = 0.95,
        stop_strings: Optional[List[str]] = None,
        force_audio_gen: bool = True,
        ras_win_len: Optional[int] = 7,
        ras_win_max_num_repeat: int = 2,
        seed: Optional[int] = None,
    ):
        """
        Stream audio generation with voice consistency using text profiles.

        This method maintains voice consistency by:
        1. Using the session's accumulated audio tokens as context
        2. Building on previous generation messages
        3. Collecting generated tokens to add to context for next generation

        Args:
            text: The text to synthesize
            session: VoiceProfileSession managing voice consistency state
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Top-p (nucleus) sampling
            stop_strings: Strings that stop generation
            force_audio_gen: Force audio token generation
            ras_win_len: RAS window length for reducing repetition
            ras_win_max_num_repeat: Max repeats in RAS window
            seed: Random seed for reproducibility

        Yields:
            HiggsAudioStreamerDelta objects with text/audio tokens
        """
        if stop_strings is None:
            stop_strings = ["<|end_of_text|>", "<|eot_id|>"]
        if ras_win_len is not None and ras_win_len <= 0:
            ras_win_len = None

        # Add user message to session
        session.add_user_message(text)

        # Build ChatML sample with accumulated context
        chat_ml_sample = ChatMLSample(messages=session.get_context_messages())

        # Get context audio IDs
        context_audio_ids = session.get_context_audio_ids()

        with torch.no_grad():
            # Prepare inputs with context
            inputs = self._prepare_inputs_with_context(
                chat_ml_sample,
                context_audio_ids,
                force_audio_gen=force_audio_gen
            )

            self._prepare_kv_caches()

            # Create streamer for async iteration
            streamer = AsyncHiggsAudioStreamer(
                self.tokenizer,
                audio_num_codebooks=self.model.config.audio_num_codebooks,
                skip_prompt=True,
            )

            # Collect audio tokens during streaming for context accumulation
            collected_audio_tokens = []

            generation_kwargs = dict(
                **inputs,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                stop_strings=stop_strings,
                tokenizer=self.tokenizer,
                do_sample=False if temperature == 0.0 else True,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                past_key_values_buckets=self.kv_caches,
                ras_win_len=ras_win_len,
                ras_win_max_num_repeat=ras_win_max_num_repeat,
                seed=seed,
                streamer=streamer,
            )

            thread = threading.Thread(target=self.model.generate, kwargs=generation_kwargs)
            thread.start()

            async for delta in streamer:
                # Collect audio tokens for session context
                if delta.audio_tokens is not None:
                    collected_audio_tokens.append(delta.audio_tokens)

                yield delta

            # After streaming completes, process collected audio tokens
            thread.join()  # Ensure generation is complete

            if collected_audio_tokens:
                # Stack and process audio tokens
                audio_tokens_tensor = torch.stack(collected_audio_tokens, dim=1)  # (num_codebooks, seq_len)

                # Revert delay pattern and clip to valid range
                processed_audio = revert_delay_pattern(audio_tokens_tensor)
                processed_audio = processed_audio.clip(0, self.audio_codebook_size - 1)[:, 1:-1]

                # Add to session context
                session.add_generated_audio(processed_audio)

    def create_voice_profile_session(
        self,
        speaker_desc: str,
        scene_prompt: Optional[str] = None,
        generation_chunk_buffer_size: Optional[int] = 3,
    ) -> VoiceProfileSession:
        """
        Create a new voice profile session for consistent voice streaming.

        Args:
            speaker_desc: Description of the speaker/voice characteristics
                e.g., "Male, American accent, moderate pitch, friendly tone, very clear audio"
            scene_prompt: Description of the audio environment
                e.g., "Audio is recorded from a quiet room."
            generation_chunk_buffer_size: Number of past generations to keep as context
                None means keep all, integer limits the buffer

        Returns:
            VoiceProfileSession configured for consistent voice generation

        Example:
            >>> session = engine.create_voice_profile_session(
            ...     speaker_desc="Female, British accent, calm tone, clear articulation",
            ...     scene_prompt="Professional podcast recording studio",
            ...     generation_chunk_buffer_size=3
            ... )
            >>> async for delta in engine.generate_delta_stream_with_voice_profile(
            ...     "Hello, welcome to the show!",
            ...     session
            ... ):
            ...     # Process streaming delta
            ...     pass
        """
        config = VoiceProfileConfig(
            speaker_desc=speaker_desc,
            scene_prompt=scene_prompt,
            generation_chunk_buffer_size=generation_chunk_buffer_size,
        )
        return VoiceProfileSession(config=config)
```

---

### Part 3: Integration Points

#### 3.1 Updates to `serve_engine.py`

**File**: `boson_multimodal/serve/serve_engine.py`

Changes required:
1. Import the new `VoiceProfileSession` and `VoiceProfileConfig`
2. Import `revert_delay_pattern` from model utils
3. Add `_prepare_inputs_with_context` method
4. Add `generate_delta_stream_with_voice_profile` async method
5. Add `create_voice_profile_session` factory method

#### 3.2 New Exports

**File**: `boson_multimodal/serve/__init__.py` (if exists, or create)

```python
from .serve_engine import (
    HiggsAudioServeEngine,
    HiggsAudioResponse,
    HiggsAudioStreamerDelta,
    AsyncHiggsAudioStreamer,
)
from .voice_profile_session import (
    VoiceProfileSession,
    VoiceProfileConfig,
)

__all__ = [
    "HiggsAudioServeEngine",
    "HiggsAudioResponse",
    "HiggsAudioStreamerDelta",
    "AsyncHiggsAudioStreamer",
    "VoiceProfileSession",
    "VoiceProfileConfig",
]
```

---

### Part 4: Usage Example

```python
import asyncio
import torch
import numpy as np
from boson_multimodal.serve import HiggsAudioServeEngine

async def main():
    # Initialize engine
    engine = HiggsAudioServeEngine(
        model_name_or_path="bosonai/higgs-audio-v2-generation-3B-base",
        audio_tokenizer_name_or_path="bosonai/higgs-audio-v2-tokenizer",
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    # Create a voice profile session
    session = engine.create_voice_profile_session(
        speaker_desc="Female, British accent, calm and professional tone, moderate pace, very clear audio",
        scene_prompt="Audio is recorded from a quiet podcast studio.",
        generation_chunk_buffer_size=3,  # Keep last 3 generations as context
    )

    # Text chunks to synthesize
    texts = [
        "Welcome to our podcast! Today we're discussing artificial intelligence.",
        "Machine learning has transformed how we interact with technology.",
        "Let's explore some fascinating developments in this field.",
    ]

    # Stream each text chunk with consistent voice
    all_audio_chunks = []

    for text in texts:
        print(f"Generating: {text}")
        audio_tokens = []

        async for delta in engine.generate_delta_stream_with_voice_profile(
            text=text,
            session=session,
            max_new_tokens=2048,
            temperature=1.0,
            top_p=0.95,
            force_audio_gen=True,
        ):
            if delta.audio_tokens is not None:
                audio_tokens.append(delta.audio_tokens)
                # Can process audio in real-time here

        if audio_tokens:
            # Decode collected audio tokens to waveform
            audio_tensor = torch.stack(audio_tokens, dim=1)
            # ... decode and accumulate audio

    print(f"Generated {session.generation_count} audio segments with consistent voice")

if __name__ == "__main__":
    asyncio.run(main())
```

---

## Key Technical Details

### Voice Profile Format

The voice profile system uses special tags:
- `<|scene_desc_start|>` and `<|scene_desc_end|>` - Wraps the scene/speaker description
- `SPEAKER0: <description>` - Format for speaker voice characteristics

Example system message:
```
Generate audio following instruction.

<|scene_desc_start|>
Audio is recorded from a quiet room.

SPEAKER0: Male, American accent, modern speaking rate, moderate-pitch, friendly tone, and very clear audio.
<|scene_desc_end|>
```

### Voice Descriptor Examples (from `profile.yaml`)

| Profile | Description |
|---------|-------------|
| `male_en` | Male, American accent, modern speaking rate, moderate-pitch, friendly tone, and very clear audio |
| `female_en_story` | She speaks with a calm, gentle, and informative tone at a measured pace, with excellent articulation and very clear audio |
| `male_en_british` | He speaks with a clear British accent and a conversational, inquisitive tone. His delivery is articulate and at a moderate pace |
| `female_en_british` | A female voice with a clear British accent speaking at a modern rate with a moderate-pitch in an expressive and friendly tone |

### Buffer Management

The `generation_chunk_buffer_size` parameter controls context accumulation:

- `None`: Keep ALL previous generations (may cause context overflow)
- `1`: Only use immediately previous generation as context
- `3` (recommended): Keep last 3 generations for good balance
- Higher values: More consistent voice but larger context/slower generation

### Audio Token Processing

Generated audio tokens must be processed before storing as context:

```python
# 1. Revert delay pattern (Higgs uses interleaved codebook delay)
processed_audio = revert_delay_pattern(audio_tokens_tensor)

# 2. Clip to valid codebook range
processed_audio = processed_audio.clip(0, codebook_size - 1)

# 3. Remove BOS/EOS tokens (first and last columns)
processed_audio = processed_audio[:, 1:-1]
```

---

## Testing Plan

### Unit Tests

1. **VoiceProfileSession Tests**
   - Test session creation with various configs
   - Test message accumulation
   - Test buffer limit enforcement
   - Test reset functionality

2. **Integration Tests**
   - Test `_prepare_inputs_with_context` with mock audio IDs
   - Test voice consistency across multiple generations
   - Test streaming delta collection

### Manual Testing

1. Generate 3+ consecutive audio clips with same session
2. Verify voice characteristics remain consistent
3. Compare with non-session streaming (should sound different)
4. Test buffer size effects on quality vs. performance

---

## Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `boson_multimodal/serve/voice_profile_session.py` | CREATE | Session state management |
| `boson_multimodal/serve/serve_engine.py` | MODIFY | Add new streaming method |
| `boson_multimodal/serve/__init__.py` | MODIFY/CREATE | Export new classes |

---

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Context overflow with large buffers | OOM errors | Use reasonable buffer size (3-5), monitor memory |
| Audio quality degradation with buffer | Poor output | Test optimal buffer sizes, allow configuration |
| Thread safety in async streaming | Race conditions | Use proper synchronization for audio collection |
| Session state corruption | Inconsistent voice | Immutable config, clear state management |

---

## Next Steps

1. ✅ Analysis complete - understand existing codebase
2. ⏳ Create `voice_profile_session.py` with session management
3. ⏳ Add `_prepare_inputs_with_context` to serve_engine
4. ⏳ Implement `generate_delta_stream_with_voice_profile` method
5. ⏳ Add `create_voice_profile_session` factory method
6. ⏳ Update exports
7. ⏳ Create usage example/test script
8. ⏳ Test voice consistency across generations

---

## Summary

This implementation enables voice-consistent streaming by:

1. **Managing Session State**: `VoiceProfileSession` tracks accumulated audio tokens and messages
2. **Building Context**: Each generation includes previous audio as context
3. **Buffer Management**: Limits context size to prevent overflow
4. **Seamless Integration**: New method mirrors existing streaming API

The approach mirrors the proven pattern from `generation.py` while adapting it for the async streaming architecture of `HiggsAudioServeEngine`.
