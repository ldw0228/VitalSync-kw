# 003 — Reverse-order synchronization

Subjects were processed in reverse order as requested. The method loads three 40 Hz UWB streams and BIOPAC RSP, detects strong chest-press protrusions, merges detections within four seconds, then searches offsets from −12 to +12 seconds at 0.1-second resolution for alignment with early radar-motion peaks. Difficult cases are reviewed rather than silently forced. The supplied single-subject interactive reference is [`code/sync_tool_S02.m`](code/sync_tool_S02.m); configure its subject, raw-data and output paths locally.
