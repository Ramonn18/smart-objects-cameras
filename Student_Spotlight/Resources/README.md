# Student Spotlight — Project Definition & Scope

---

## Project Overview

<!-- What is this project? 1–3 sentences describing the core idea. -->
This is a Smart classroom project. The overal concept is to create a student spotlight by utilizing code and avilable sofware such as lights and video cameras/ai video cameras that can determine when a student want to participate and spotlight them into queue by utilizing common gestures significant to participation.

---

## Problem Statement

<!-- What problem does this solve? Who experiences it? Why does it matter? -->
The class has had a participation problems, this project is trying to increase student participation by making it fun and responcive to body gestures. Some students seem to have curiosities but don't tent to participate, in here we are designing a spotlight tool that will give them the "want" to participate
---

## Goals

<!-- What should this project accomplish? Use bullet points. -->

-Increase student participation
-Student notice of participation
-Student Queue for participation
-Advice students that there are not bad questions

---

## Scope

### In Scope

- Gesture detection via OAK-D camera (e.g. raised hand) to enter the participation queue
- Sound cue played on the Raspberry Pi to acknowledge a student joining the queue (room-level feedback)
- Discord DM sent to the student confirming their queue position ("You're #2 — you'll be called on soon")
- Participation queue management (add, remove, display current order)
- Teacher-facing queue view via Discord channel or bot command

### Out of Scope

- Controllable physical spotlight or directional lighting hardware
- Face recognition or student identity tracking by appearance
- Persistent session history or grade/participation analytics
- Projector overlay or screen-based visual display (future consideration)

---

## Users / Audience

- **Students** — raise their hand (gesture) to join the queue; receive a private Discord DM confirming their spot
- **Teacher** — sees the live queue via Discord; calls on students in order; dismisses them from the queue
- **Classroom (ambient)** — hears the chime when a student joins, giving the room a shared moment of acknowledgment

---

## Key Features

| Feature | Description | Priority |
|---------|-------------|----------|
| Gesture detection | OAK-D detects a raised hand or participation gesture | High |
| Sound cue | Pi plays a chime when a student enters the queue | High |
| Discord DM | Student receives a private message with their queue position | High |
| Queue management | Students are added/removed in order; teacher controls via bot command | High |
| Teacher queue view | `!queue` command shows who's waiting and in what order | Medium |
| Queue clearing | `!next` or `!clear` command to dismiss a student or reset the queue | Medium |

---

## Hardware

- **Camera:** Luxonis OAK-D (connected to Raspberry Pi)
- **Raspberry Pi:** orbit / gravity / horizon
- **Audio output:** Pi's 3.5mm audio jack or HDMI audio → speaker or monitor for chime playback
- **Display / Projection:** Not required for this scope (future consideration)

---

## Technical Approach

- **Detection model:** YOLO v6 (person/pose) or MediaPipe pose estimation — detect raised arm/hand gesture
- **Communication:** Discord bot (discord.py) — sends DMs to students, posts queue updates to class channel
- **Output / feedback:** `pygame` or `aplay` on Pi for chime playback; Discord DM for personal confirmation
- **Queue state:** In-memory queue in the bot, optionally backed by a JSON file for persistence across restarts

---

## Open Questions

- What exact gesture triggers a queue entry? (full raised hand, any upward arm motion?)
- How does the system know *which* student gestured — do students need Discord accounts linked to a seat/camera zone?
- Does the teacher dismiss students manually (`!next`) or does the system auto-remove after a timeout?
- Which Pi / camera covers which part of the room — one camera per zone, or one camera for the whole room?
- What sound plays for the chime — a simple beep, a tone, or a short audio clip?

---

## References & Inspiration

<!-- Links, images, prior art, or examples that inspired this project. -->

-

---

*Last updated: 2026-03-23 — scope narrowed to sound cue + Discord DM outputs*
