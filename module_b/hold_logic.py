"""
Module B  —  Step B4: Hold Logic (Flicker Prevention)
======================================================
TASK
----
Prevent the HUD departure warning from flickering off and on rapidly
when YOLO momentarily misses a lane line for 1–2 frames.

WHY THIS IS NEEDED
------------------
Even on a straight, stable road, YOLO's segmentation output can vary
slightly between consecutive frames:
  - A lane line near the edge of the frame may drop out for 1–2 frames
  - The polynomial fit becomes unavailable → offset is None → raw state = CENTERED
  - Without hold logic: the warning appears  →  disappears  →  reappears
    within a fraction of a second. This is confusing and unsafe.

ALGORITHM  (Section B4 of the implementation plan)
---------------------------------------------------
    Constant:
      HOLD_FRAMES = 6   ← frames to hold active state after signal clears

    Per frame:
      IF new_state != CENTERED:       ← an active warning is detected
          held_state   = new_state    ← adopt it and (re)start the timer
          hold_counter = HOLD_FRAMES

      ELSE:  (B3 returned CENTERED this frame)
          IF hold_counter > 0:
              hold_counter -= 1
              # held_state unchanged → keep showing the previous warning
          ELSE:
              held_state = CENTERED   ← counter exhausted → clear HUD

    Output to HUD: held_state  (NOT the raw new_state from B3)

GUARANTEE
---------
    Every active departure warning is displayed for at least
    HOLD_FRAMES / FPS  seconds after the triggering condition clears.
    At 30 fps: 6 frames ≈ 0.2 s (barely visible pause, prevents flicker).

INPUTS
------
  new_state : str  — raw departure state from B3 (classify_departure)

OUTPUTS
-------
  held_state : str  — the state to send to the HUD this frame

USAGE
-----
    holder = DepartureHoldLogic()

    for each frame:
        raw_state = classify_departure(...)
        display_state = holder.update(raw_state)
"""

from .departure_classifier import CENTERED, ACTIVE_STATES


# ── Tuning constant ───────────────────────────────────────────────────────
# Number of frames to hold an active departure state after the raw signal
# returns to CENTERED. Matches the plan's Tuning Reference: HOLD_FRAMES = 6.
HOLD_FRAMES = 6
# ──────────────────────────────────────────────────────────────────────────


class DepartureHoldLogic:
    """
    Hysteresis hold for departure warning states.

    Keeps an active warning visible for HOLD_FRAMES frames after the
    raw B3 signal clears, preventing single-frame flicker.

    Usage
    -----
        holder = DepartureHoldLogic()

        for frame in video:
            raw_state    = classify_departure(...)
            display_state = holder.update(raw_state)
            # use display_state for the HUD, not raw_state
    """

    def __init__(self, hold_frames: int = HOLD_FRAMES):
        """
        Parameters
        ----------
        hold_frames : int
            How many CENTERED frames to wait before clearing the warning.
            Plan default: 6.
        """
        self._hold_frames:  int = hold_frames
        self._held_state:   str = CENTERED
        self._hold_counter: int = 0

    # ─────────────────────────────────────────────────────────────────────
    def update(self, new_state: str) -> str:
        """
        Feed the raw B3 state and get the held state for the HUD.

        Parameters
        ----------
        new_state : str
            Raw departure state from classify_departure() this frame.

        Returns
        -------
        str : the departure state to display on the HUD.
              Equals new_state when active, or the last active state
              while the hold counter is still running.
        """
        if new_state in ACTIVE_STATES:
            # Active warning detected — adopt immediately, reset timer
            self._held_state   = new_state
            self._hold_counter = self._hold_frames
        else:
            # Raw state is CENTERED
            if self._hold_counter > 0:
                self._hold_counter -= 1
                # self._held_state unchanged → keep showing previous warning
            else:
                self._held_state = CENTERED

        return self._held_state

    # ─────────────────────────────────────────────────────────────────────
    def reset(self):
        """
        Clear hold state (e.g. when switching video clips).
        """
        self._held_state   = CENTERED
        self._hold_counter = 0

    # ─────────────────────────────────────────────────────────────────────
    @property
    def current(self) -> str:
        """Return the last held state without updating."""
        return self._held_state

    @property
    def counter(self) -> int:
        """Return the remaining hold frames (0 = no active hold)."""
        return self._hold_counter
