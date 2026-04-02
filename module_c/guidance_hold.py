"""
Module C  —  Step C7: Guidance Hysteresis Hold
===============================================
TASK
----
Prevent the guidance banner from flickering on and off rapidly when YOLO
momentarily misses a vehicle detection for 1–2 frames.

WHY THIS IS NEEDED
------------------
YOLO confidence oscillates frame-to-frame:
  - A vehicle near the proximity threshold may appear CLOSE one frame,
    then FAR the next, then CLOSE again.
  - Without hold logic: the guidance banner appears → disappears →
    reappears within fractions of a second.
  - This is distracting and reduces driver trust in the system.

ALGORITHM  (Section C7 of the implementation plan)
---------------------------------------------------
    Constant:
      GUIDE_HOLD_FRAMES = 8   ← frames to hold active guidance after
                                 the raw signal returns to GUIDE_NONE

    Per frame:
      IF new_state != GUIDE_NONE:     ← guidance signal is active
          held_state   = new_state    ← adopt immediately
          hold_counter = GUIDE_HOLD_FRAMES   ← reset timer

      ELSE:  (raw state is GUIDE_NONE this frame)
          IF hold_counter > 0:
              hold_counter -= 1
              # held_state unchanged → keep showing previous guidance
          ELSE:
              held_state = GUIDE_NONE   ← counter exhausted → clear banner

      Output to HUD: held_state  (not raw new_state)

DIFFERENCE FROM MODULE B HOLD
------------------------------
Module B (departure) holds for 6 frames (HOLD_FRAMES = 6).
Module C (guidance)  holds for 8 frames (GUIDE_HOLD_FRAMES = 8).

Guidance holds longer because:
  - Vehicle detections flicker more than lane line detections
  - Guidance banners are advisory → staying slightly longer is safer
    than flashing on/off

NOTE ON GUIDE_URGENT
--------------------
GUIDE_URGENT is also held like any other active guidance state.
This means even if the front vehicle briefly drops below the
VERY_CLOSE threshold for 1–2 frames, the urgent brake warning
stays visible for 8 more frames.

INPUTS
------
  new_state : str  — raw guidance state from decide_guidance()

OUTPUT
------
  held_state : str  — the state to display on the HUD this frame
"""

from .guidance_states import GUIDE_NONE, ACTIVE_GUIDE_STATES


# ── Tuning constant ───────────────────────────────────────────────────────
# Frames to hold an active guidance state after raw signal returns to NONE.
# Plan Tuning Reference: GUIDE_HOLD_FRAMES = 8
GUIDE_HOLD_FRAMES = 8
# ──────────────────────────────────────────────────────────────────────────


class GuidanceHoldLogic:
    """
    Hysteresis hold for guidance output states.

    Keeps an active guidance banner visible for GUIDE_HOLD_FRAMES frames
    after the raw C4 signal clears, preventing single-frame flicker.

    Usage
    -----
        holder = GuidanceHoldLogic()

        for frame in video:
            raw_state     = decide_guidance(...)
            display_state = holder.update(raw_state)
            # use display_state for the HUD banner
    """

    def __init__(self, hold_frames: int = GUIDE_HOLD_FRAMES):
        """
        Parameters
        ----------
        hold_frames : int
            How many GUIDE_NONE frames to wait before clearing the banner.
            Plan default: 8.
        """
        self._hold_frames:  int = hold_frames
        self._held_state:   str = GUIDE_NONE
        self._hold_counter: int = 0

    # ─────────────────────────────────────────────────────────────────────
    def update(self, new_state: str) -> str:
        """
        Feed the raw C4 guidance state and return the held state for HUD.

        Parameters
        ----------
        new_state : str
            Raw guidance state from decide_guidance() this frame.

        Returns
        -------
        str : the guidance state to display on the HUD.
        """
        if new_state in ACTIVE_GUIDE_STATES:
            # Active guidance: adopt immediately and reset hold timer
            self._held_state   = new_state
            self._hold_counter = self._hold_frames
        else:
            # Raw state is GUIDE_NONE
            if self._hold_counter > 0:
                self._hold_counter -= 1
                # self._held_state unchanged → keep showing previous guidance
            else:
                self._held_state = GUIDE_NONE

        return self._held_state

    # ─────────────────────────────────────────────────────────────────────
    def reset(self):
        """Clear hold state (e.g. when switching video clips)."""
        self._held_state   = GUIDE_NONE
        self._hold_counter = 0

    # ─────────────────────────────────────────────────────────────────────
    @property
    def current(self) -> str:
        """Return the last held state without updating."""
        return self._held_state

    @property
    def counter(self) -> int:
        """Return remaining hold frames (0 = no active hold)."""
        return self._hold_counter
