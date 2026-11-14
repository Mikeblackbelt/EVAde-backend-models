"""Starting values from lit review/planning/implementation-pipeline.md (design assumptions)."""

FS_TARGET = 250.0           # Hz, common grid for ECG and EEG
ECG_BAND = (0.5, 40.0)      # Hz band-pass
EEG_BAND = (0.5, 45.0)      # Hz band-pass
NOTCH_HZ = 60.0             # mains (US data); 50 Hz for European sets via argument
ECG_WIN_S, ECG_STEP_S = 30.0, 15.0     # rhythm windows (50% overlap)
EEG_WIN_S, EEG_STEP_S = 4.0, 2.0       # EEG model windows (50% overlap)
SLEEP_EPOCH_S = 30.0
QUALITY_MIN = 0.6           # windows below this are kept but flagged low_quality

# seizure labelling (design assumptions)
SZ_MIN_OVERLAP_S = 2.0      # a 4 s window is ictal if at least 2 s overlap a seizure
SZ_PRE_BUFFER_S = 60.0      # non-ictal windows this close before onset are excluded (-1)
SZ_POST_BUFFER_S = 120.0    # ... and this close after offset

N_FOLDS = 5

# fixed 18-channel bipolar "double banana" montage, anode-cathode
BIPOLAR_PAIRS = [
    ("FP1", "F7"), ("F7", "T7"), ("T7", "P7"), ("P7", "O1"),
    ("FP1", "F3"), ("F3", "C3"), ("C3", "P3"), ("P3", "O1"),
    ("FP2", "F4"), ("F4", "C4"), ("C4", "P4"), ("P4", "O2"),
    ("FP2", "F8"), ("F8", "T8"), ("T8", "P8"), ("P8", "O2"),
    ("FZ", "CZ"), ("CZ", "PZ"),
]
BIPOLAR_NAMES = [f"{a}-{b}" for a, b in BIPOLAR_PAIRS]

# old to new 10-20 names
ELECTRODE_ALIASES = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}

# schema of every window table (see integration-and-evaluation.md section 2)
SCHEMA_COLUMNS = [
    "participant_id", "record_id", "dataset", "modality", "fs",
    "window_index", "window_start_s", "window_length_s",
    "timestamp_utc", "quality_index", "low_quality", "label", "fold",
    "pipeline_version",
]
