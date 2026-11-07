# Retry wrapper for PyBullet's GUI connection.
#
# In GUI mode a PyBullet call occasionally fails on its own ("getJointState failed.",
# "getLinkState failed.", "Error resetting visual shape info", ...) while the connection is still
# alive. Measured here - roughly 1 failure per 400,000 query calls, independent of memory use or of
# what the robot is doing, and retrying the same call succeeds right away. A painting run makes
# millions of such calls, so an uncaught failure ends the run at a random point.
#
# install() wraps the pybullet functions this project uses so that a failed call is retried a few
# times instead of raising. It does nothing in DIRECT (headless) mode, which does not have the
# problem, and it never retries once the connection is gone (window closed), so a real disconnect
# still raises immediately.
import functools
import time

import pybullet as p

# Queries and idempotent setters - repeating one of these after a failed attempt is harmless.
RETRIED = (
    "getJointStates", "getJointState", "getLinkState", "getLinkStates",
    "getBasePositionAndOrientation", "getBaseVelocity", "getNumJoints", "getJointInfo", "getBodyInfo",
    "getContactPoints", "getClosestPoints", "rayTest", "rayTestBatch",
    "resetJointState", "resetBasePositionAndOrientation",
    "setJointMotorControl2", "setJointMotorControlArray",
    "changeVisualShape", "createVisualShape", "loadTexture",
    "getKeyboardEvents", "getDebugVisualizerCamera", "resetDebugVisualizerCamera",
)

recovered = 0  # how many transient failures were absorbed (printed at the end of a run)


def _wrap(fn, retries, delay):
    @functools.wraps(fn)
    def call(*args, **kwargs):
        global recovered
        for attempt in range(retries + 1):
            try:
                return fn(*args, **kwargs)
            except p.error:
                if attempt == retries or not p.isConnected():
                    raise
                recovered += 1
                time.sleep(delay)
    return call


def install(retries: int = 3, delay: float = 0.02):
    if getattr(p, "_retry_installed", False):
        return
    for name in RETRIED:
        fn = getattr(p, name, None)
        if fn is not None:
            setattr(p, name, _wrap(fn, retries, delay))
    p._retry_installed = True
