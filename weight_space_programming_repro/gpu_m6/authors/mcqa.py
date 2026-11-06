"""Stand-in for the authors' unreleased weightenv/lmo/tasks/mcqa.py. mathqa.py imports only loop_signals from it, and
only for its loop diagnostics (mathqa.loop_signals), which the M6 evaluation does not use. Grading (extract,
is_correct) does not touch it."""


def loop_signals(text):
    return {"repetitive_end": None}
