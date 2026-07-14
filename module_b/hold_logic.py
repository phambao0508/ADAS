from .departure_classifier import CENTERED, ACTIVE_STATES

HOLD_FRAMES = 6

class DepartureHoldLogic:

    def __init__(self, hold_frames: int = HOLD_FRAMES):

        self._hold_frames:  int = hold_frames
        self._held_state:   str = CENTERED
        self._hold_counter: int = 0

    def update(self, new_state: str) -> str:

        if new_state in ACTIVE_STATES:

            self._held_state   = new_state
            self._hold_counter = self._hold_frames
        else:

            if self._hold_counter > 0:
                self._hold_counter -= 1

            else:
                self._held_state = CENTERED

        return self._held_state

    def reset(self):

        self._held_state   = CENTERED
        self._hold_counter = 0

    @property
    def current(self) -> str:

        return self._held_state

    @property
    def counter(self) -> int:

        return self._hold_counter
