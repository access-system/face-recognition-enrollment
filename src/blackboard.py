import threading
from typing import Dict, Any, Optional
from enum import Enum


class FrameState(Enum):
    # States for different stages of processing
    DEFAULT_FRAME = 'default_frame'
    PROCESSED_FRAME = 'processed_frame'
    DETECTED_FACE = 'detected_face'
    VALIDATED_FACE = 'validated_face'
    ALIGNED_FACE = 'aligned_face'
    EMBEDDING = 'embedding'

    # Messages
    LAST_INFO_MSG = 'last_info_msg'
    LAST_ERROR_MSG = 'last_error_msg'


class BlackboardStateful:
    _shared_state = None
    _lock = threading.Lock()

    def __init__(self):
        if BlackboardStateful._shared_state is None:
            with BlackboardStateful._lock:
                if BlackboardStateful._shared_state is None:
                    BlackboardStateful._shared_state = {
                        FrameState.DEFAULT_FRAME.value: None,
                        FrameState.PROCESSED_FRAME.value: None,
                        FrameState.DETECTED_FACE.value: None,
                        FrameState.VALIDATED_FACE.value: None,
                        FrameState.ALIGNED_FACE.value: None,
                        FrameState.EMBEDDING.value: None,

                        FrameState.LAST_INFO_MSG.value: None,
                        FrameState.LAST_ERROR_MSG.value: None,
                    }

        self._state: Dict[str, Any] = BlackboardStateful._shared_state

    def set_state(self, key: str, value: Any) -> None:
        with BlackboardStateful._lock:
            if key in self._state:
                self._state[key] = value

    def get_state(self, key: str) -> Optional[Any]:
        with BlackboardStateful._lock:
            return self._state.get(key)

    def has_state(self, key: str) -> bool:
        with BlackboardStateful._lock:
            return self._state.get(key) is not None

    def reset_state(self, key: str) -> None:
        with BlackboardStateful._lock:
            if key in self._state:
                self._state[key] = None

    def reset_all(self) -> None:
        with BlackboardStateful._lock:
            self._state = {k: None for k in self._state}
