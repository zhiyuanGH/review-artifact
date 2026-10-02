"""Workflow planning stage helpers.

The planning stage owns logical workflow generation only. It writes artifacts
under <run-dir>/planning and leaves backend assignment/compilation to the
deployment stage.
"""

PLANNING_STAGE_DIR = "planning"

