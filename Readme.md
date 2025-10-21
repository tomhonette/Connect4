# Connect 4 – AI Project

## Overview
This project is an implementation of **Connect 4** featuring multiple gameplay modes, including an **AI opponent** powered by a **Minimax algorithm** with jitter and random bias for more natural gameplay variation.  

The project was developed as part of a **second-year “Introduction to AI” course**.  
All UI and core game functionalities were provided — only the **AI logic (Minimax algorithm)** was implemented by me.

---

## How to Play

### 1. Play Against the AI
Run the following command to start a game against the AI:
```bash
  python connect4_ai1.py
```

### 2. Play Against Another Player
Run this command to play locally with another real player:
```bash
  python connect4.py
```

### 3. Watch the AI Compete itself
Let the AI face itself by running:
```bash
  python connect4_ai2.py
```

### AI Description
The AI opponent uses a Minimax algorithm enhanced with:
- A **jitter effect**, adding slight randomness to prevent deterministic behavior.
- A **small random bias**, introducing diverse decision patterns.

These adjustments make the AI feel more human and **less predictable**, while still **playing competitively**.

### Project Structure

- connect4.py  # Player vs Player
- connect4_ai1.py # Player vs AI
- connect4_ai2.py # AI vs AI
- aiplayer.py # AI implementation

### Credits
- Author (AI implementation): **Tom Honette**
- Course: **Introduction to Artificial Intelligence** - Year 2, Period 1
- Provided by: **University teaching staff** (UI and game engine)