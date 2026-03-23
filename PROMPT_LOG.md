# Prompt Log

A running log of prompts and responses for this project.

---

## Session: 2026-03-16

---

### Prompt 1
**User:** In this folder create a log (documentation) of my prompts and the prompt responses

**Response:** Created this file (`PROMPT_LOG.md`) in the project root to serve as a running log of all prompts and responses. Each entry is dated and numbered within its session.

---

### Prompt 2
**User:** Do that every time automatically

**Response:** Understood. From now on, every prompt and response will be automatically appended to this log file.

---

### Prompt 3
**User:** Create a sub folder in this folder title E-Wall_Brain_Storm

**Response:** Created the subfolder `E-Wall_Brain_Storm` inside the project root.

---

### Prompt 4
**User:** Base on the E-Wall_Brain_Storm folder file. Lets create an Initial .py file for the project

**Response:** Read the brainstorm image (`Smart Classroom - ixD 27'.png`) to understand the E-Wall (E-CA) concept — an ultra-short-throw projector + OAK-D camera system that tracks presenter gestures on stage to drive dynamic visual projections. Created `E-Wall_Brain_Storm/e_wall.py` with: YOLO person detection via DepthAI 3.x pipeline, stage zone calibration, gesture classification (center_stage, left_wing, right_wing, close_to_wall, back_of_stage), debouncing, Discord notifications, live display with zone overlay, and status file output.

---

## Session: 2026-03-23

---

### Prompt 1
**User:** Base on the Student_Spotlight folder create an initial .py folder for the project

**Response:** Explored the (empty) `Student_Spotlight` folder, then reviewed the broader project for context. Created `Student_Spotlight/student_spotlight.py` — a classroom presenter recognition system using OAK-D + YOLO that detects when a student enters a configurable spotlight zone, sends Discord announcements on entry/exit with duration tracking, debounces state changes (2s), shows a live display with zone overlay, saves periodic screenshots, and writes a status JSON file. Follows the same CLI and architectural patterns as `e_wall.py` and `person_detector.py`.

---

### Prompt 2
**User:** dont forget to start the Prompt log everytime we begin a session

**Response:** Noted. Added today's session header and backfilled this session's entries in PROMPT_LOG.md. The feedback memory already exists and will be applied going forward — every session will begin with a new dated header in the log.

---

### Prompt 3
**User:** now lets start the student spotlight project: in the Student_Spotlight/Resources let create a read me where we will start the Project definition/Scope

**Response:** Created `Student_Spotlight/Resources/README.md` with a structured project definition template covering: Project Overview, Problem Statement, Goals, Scope (in/out), Users/Audience, Key Features table, Hardware, Technical Approach, Open Questions, and References sections — ready to be filled in as the project is defined.

---
