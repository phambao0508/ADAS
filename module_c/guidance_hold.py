from .guidance_states import GUIDE_NONE, ACTIVE_GUIDE_STATES

GUIDE_HOLD_FRAMES = 8

class GuidanceHoldLogic:

    def __init__(self, hold_frames: int = GUIDE_HOLD_FRAMES):

        self._hold_frames:  int = hold_frames
        self._held_state:   str = GUIDE_NONE
        self._hold_counter: int = 0

    def update(self, new_state: str) -> str:

        if new_state in ACTIVE_GUIDE_STATES:

            self._held_state   = new_state
            self._hold_counter = self._hold_frames
        else:

            if self._hold_counter > 0:
                self._hold_counter -= 1

            else:
                self._held_state = GUIDE_NONE

        return self._held_state

    def reset(self):

        self._held_state   = GUIDE_NONE
        self._hold_counter = 0

    @property
    def current(self) -> str:

        return self._held_state

    @property
    def counter(self) -> int:

        return self._hold_counter
