import os

import torch
import soundfile
import numpy as np
from loguru import logger

from boson_multimodal.serve.serve_engine import (
    HiggsAudioServeEngine,
)
from boson_multimodal.data_types import ChatMLSample, Message


def revert_delay_pattern(data, start_idx=0):
    """Convert samples encoded with delay pattern back to the original form.

    Args:
        data (:obj:`torch.Tensor`):
            The data with delay pattern applied. It will have shape (num_codebooks, seq_len + num_codebooks - 1).

    Returns:
        ret (:obj:`torch.Tensor`):
            Recovered data with delay pattern removed. It will have shape (num_codebooks, seq_len).
    """
    assert len(data.shape) == 2
    assert data.shape[1] - data.shape[0] >= start_idx
    out_l = []
    num_codebooks = data.shape[0]
    for i in range(num_codebooks):
        out_l.append(data[i : (i + 1), i + start_idx : (data.shape[1] - num_codebooks + 1 + i)])
    return torch.cat(out_l, dim=0)


async def audio_generate_stream(**kwargs):
    ASSETS_DIR="./assets"
    HF_MODEL_DIR = "./ckpt"
    MODEL_PATH = os.getenv("LLM_MODEL", "bosonai/higgs-audio-v2-generation-3B-base")
    AUDIO_TOKENIZER_PATH = os.getenv("AUDIO_TOKENIZER_PATH", "bosonai/higgs-audio-v2-tokenizer")
    model_path = os.path.join(HF_MODEL_DIR, MODEL_PATH)
    audio_tokenizer_path = os.path.join(HF_MODEL_DIR, AUDIO_TOKENIZER_PATH)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    serve_engine = HiggsAudioServeEngine(
        model_path,
        audio_tokenizer_path,
        device=device,
        torch_dtype=torch.bfloat16,
    )

    system_prompt = "Generate audio following instruction.\n\n<|scene_desc_start|>\nAudio is recorded from a quiet room.\n<|scene_desc_end|>"

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
    for i in range(1):
        logger.info(f"{i} Starting generation...")
        # output = serve_engine.generate_stream(
        output = serve_engine.generate_delta_stream(
            chat_ml_sample=ChatMLSample(messages=messages),
            max_new_tokens=1024,
            temperature=0.3,
            top_p=0.95,
            top_k=50,
            stop_strings=["<|end_of_text|>", "<|eot_id|>"],
        )

        audio_tokens = []
        audio_tensor = None
        CHUNK_SIZE = 16
        seq_len = 0
        waveform_list = []

        chunk_overlap_duration = kwargs.get("chunk_overlap_duration", 0.04)
        cross_fade_samples = int(
            chunk_overlap_duration * serve_engine.audio_tokenizer.sampling_rate
        )
        fade_out = np.linspace(1, 0, cross_fade_samples)
        fade_in = np.linspace(0, 1, cross_fade_samples)

        with torch.inference_mode():
            async for delta in output:
                if delta.text:
                    print(delta.text, end="", flush=True)

                if delta.audio_tokens is None:
                    continue

                if torch.all(delta.audio_tokens == 1025):
                    break

                # print(f"{delta.audio_tokens=}")
                audio_tokens.append(delta.audio_tokens[:, None])
                audio_tensor = torch.cat(audio_tokens, dim=-1)
                print(f"{audio_tensor=}")

                if torch.all(delta.audio_tokens != 1024):
                    seq_len += 1
                    print(f"{delta.audio_tokens=} {seq_len=}")
                if seq_len > 0 and seq_len % CHUNK_SIZE == 0:
                    vq_code = (
                        revert_delay_pattern(audio_tensor, start_idx=seq_len - CHUNK_SIZE + 1)
                        .clip(0, 1023)
                        .to(device)
                    )

                    print(f"vq_code shape: {vq_code.shape} {vq_code=}")
                    waveform_numpy = serve_engine.audio_tokenizer.decode(vq_code.unsqueeze(0))[0, 0]
                    waveform_list.append(waveform_numpy)

                    # gen_audio_path = os.path.join(ASSETS_DIR, f"higgsv2_gen_audio_{i}_{seq_len}.wav")
                    # soundfile.write(gen_audio_path, waveform_numpy, serve_engine.audio_tokenizer.sampling_rate)
                    # info = soundfile.info(gen_audio_path, verbose=True)
                    # print(f"\nSaved audio chunk to {gen_audio_path}, duration: {info.duration:.2f} seconds")

            if seq_len > 0 and seq_len % CHUNK_SIZE != 0 and audio_tensor is not None:
                print(f"{audio_tensor=} {seq_len=}")
                vq_code = (
                    revert_delay_pattern(audio_tensor, start_idx=seq_len - seq_len % CHUNK_SIZE + 1)
                    .clip(0, 1023)
                    .to(device)
                )

                print(f"vq_code shape: {vq_code.shape} {vq_code=}")
                waveform_numpy = serve_engine.audio_tokenizer.decode(vq_code.unsqueeze(0))[0, 0]
                waveform_list.append(waveform_numpy)

            # new_audio = np.concatenate(waveform_list)

            for j, audio in enumerate(waveform_list):
                if j == 0:
                    new_audio = audio[:-cross_fade_samples]
                else:
                    cross_faded_overlap = (
                        waveform_list[j - 1][-cross_fade_samples:] * fade_out
                        + audio[:cross_fade_samples] * fade_in
                    )
                    new_audio = np.concatenate(
                        [
                            new_audio,
                            cross_faded_overlap,
                            audio[cross_fade_samples:-cross_fade_samples],
                        ]
                    )
            new_audio = np.concatenate([new_audio, audio[-cross_fade_samples:]])

            gen_audio_path = os.path.join(ASSETS_DIR, f"higgsv2_gen_audio_stream_{i}.wav")
            soundfile.write(
                gen_audio_path, new_audio, serve_engine.audio_tokenizer.sampling_rate, "PCM_16"
            )
            info = soundfile.info(gen_audio_path, verbose=True)
            print(info)
            print(f"\nSaved audio chunk to {gen_audio_path}, duration: {info.duration:.2f} seconds")