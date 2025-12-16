"""Voice profile session management for consistent voice streaming."""

from dataclasses import dataclass, field
from typing import List, Optional
import torch

from ..data_types import Message, AudioContent, ChatMLSample


@dataclass
class VoiceProfileConfig:
    """Configuration for a voice profile session.

    Attributes:
        speaker_desc: Description of the speaker/voice characteristics.
            e.g., "Male, American accent, moderate pitch, friendly tone, very clear audio"
        scene_prompt: Description of the audio environment (optional).
            e.g., "Audio is recorded from a quiet room."
        generation_chunk_buffer_size: Number of past generations to keep as context.
            - None: Keep all generations (may cause context overflow for long sessions)
            - 2 (default): Keep last 2 generations - recommended balance of consistency and efficiency
            - Higher values: More context but slower generation
    """

    speaker_desc: str
    scene_prompt: Optional[str] = None
    generation_chunk_buffer_size: Optional[int] = 2


@dataclass
class VoiceProfileSession:
    """Manages state for voice-consistent generation across multiple streaming calls.

    This class tracks accumulated audio tokens and conversation messages from
    previous generations, enabling the model to maintain consistent voice
    characteristics across multiple streaming calls.

    The session follows the pattern from HiggsAudioModelClient.generate() where:
    1. Each generation adds to the accumulated context
    2. Context includes both audio tokens and message history
    3. Buffer size limits prevent context from growing unbounded

    Example:
        >>> config = VoiceProfileConfig(
        ...     speaker_desc="Female, British accent, calm tone",
        ...     scene_prompt="Quiet studio recording",
        ...     generation_chunk_buffer_size=2
        ... )
        >>> session = VoiceProfileSession(config=config)
        >>> # Session is now ready for use with generate_delta_stream_with_voice_profile
    """

    config: VoiceProfileConfig

    # Accumulated state from previous generations
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
        """Build the system message with voice profile.

        Creates a system message in the format:
            Generate audio following instruction.

            <|scene_desc_start|>
            [scene_prompt if provided]

            SPEAKER0: [speaker_desc]
            <|scene_desc_end|>
        """
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
        """Get all messages including base and accumulated generation messages.

        Returns:
            List of messages: [system_message, user1, assistant1, user2, assistant2, ...]
        """
        return self.base_messages + self.generation_messages

    def get_context_audio_ids(self) -> List[torch.Tensor]:
        """Get accumulated audio IDs for context.

        Returns:
            List of audio token tensors from previous generations.
            Each tensor has shape (num_codebooks, seq_len).
        """
        return self.generated_audio_ids.copy()

    def add_user_message(self, text: str):
        """Add a user message to the generation context.

        Args:
            text: The text content for the user message.
        """
        self.generation_messages.append(Message(role="user", content=text))

    def add_generated_audio(self, audio_ids: torch.Tensor):
        """Add generated audio tokens and assistant message to context.

        This method should be called after audio generation completes to
        update the session with the new audio tokens.

        Args:
            audio_ids: Generated audio token tensor of shape (num_codebooks, seq_len).
                       Should already have delay pattern reverted and be clipped to valid range.
        """
        self.generated_audio_ids.append(audio_ids)
        self.generation_messages.append(
            Message(role="assistant", content=AudioContent(audio_url=""))
        )
        self.generation_count += 1

        # Apply buffer management
        self._apply_buffer_limit()

    def _apply_buffer_limit(self):
        """Trim context to buffer size limit.

        Removes oldest generations when buffer size is exceeded.
        Messages are trimmed in pairs (user + assistant) to maintain consistency.
        """
        buffer_size = self.config.generation_chunk_buffer_size
        if buffer_size is not None and len(self.generated_audio_ids) > buffer_size:
            self.generated_audio_ids = self.generated_audio_ids[-buffer_size:]
            # Each generation adds 2 messages (user + assistant)
            self.generation_messages = self.generation_messages[(-2 * buffer_size):]

    def reset(self):
        """Reset the session state, keeping the voice profile configuration.

        Clears all accumulated audio tokens and messages while preserving
        the base voice profile configuration.
        """
        self.generated_audio_ids = []
        self.generation_messages = []
        self.generation_count = 0

    @property
    def has_context(self) -> bool:
        """Check if session has accumulated context from previous generations."""
        return len(self.generated_audio_ids) > 0

    @property
    def context_size(self) -> int:
        """Get the number of generations currently in context."""
        return len(self.generated_audio_ids)
