import os

ROOT_DIR = os.path.abspath(
    os.path.dirname(os.path.dirname(__file__)),
)

DATA_ORG_DIR = os.path.join(ROOT_DIR, "data-org")
DATA_ADD_DIR = os.path.join(ROOT_DIR, "data-add")
DATA_PROC_DIR = os.path.join(ROOT_DIR, "data-proc")

STATS_DIR = os.path.join(ROOT_DIR, "statistics")

SUBMISSION_DIR = os.path.join(ROOT_DIR, "submissions")

