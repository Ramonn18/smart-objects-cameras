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

### Prompt 4
**User:** can you git merge upstream/main

**Response:** Fetched upstream (kandizzy/smart-objects-cameras) and merged into main. 73 files changed, 16,935 insertions. New additions include: classroom-api/, v-jepa/ (V-JEPA video model pipeline), pc-testing/ scripts, robot_boundary_monitor.py, docs/ slides and new markdown docs, tasks/lessons.md, and updates to CLAUDE.md, discord_bot.py, person_detector.py, and whiteboard readers. No conflicts — Student_Spotlight/ work untouched.

---

### Prompt 5
**User:** I have now written about Project Overview, Problem Statement, and Goals

**Response:** Read the updated README.md. Acknowledged the written content — problem (participation anxiety) and gesture-based queue concept are well defined. Offered observations on strengths and suggested the natural next sections to fill: Scope (in/out), Users/Audience, Key Features, and Open Questions — with specific prompts drawn from what was already written.

---

### Prompt 6
**User:** stop

**Response:** Stopped mid-task as requested.

---

## Session: 2026-03-23 (continued)

---

### Prompt 1
**User:** start

**Response:** Started new session block and began logging.

---

### Prompt 2
**User:** now this being the base of the project lets take a minute and narrow down the scope. there are things we dont have hardware wise, and that is control over where the light points. what other solution can we have for this?

**Response:** Identified available hardware (OAK-D, Pis, Discord, projector) and proposed 5 alternatives to a controllable spotlight: (1) Discord as the stage — public bot post with queue position and snapshot, (2) Projector overlay — display queue on wall/screen, (3) Sound cue — chime on gesture detection, (4) Discord DM — private queue confirmation to student, (5) USB LED strip — color changes to reflect queue state. Recommended Discord + projector overlay as the strongest fit.

---

### Prompt 3
**User:** keep the prompt log automatically updated

**Response:** Noted — logging this and all future prompts/responses going forward without being reminded.

---
