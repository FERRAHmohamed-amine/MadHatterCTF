# 🎩 Mad Hatter CTF - AI Prompt Injection Game

An interactive Capture The Flag (CTF) mini-game where players must use advanced Prompt Injection techniques to trick a local AI into revealing a secret code.

## 🧠 About the Project
This project demonstrates **Defense in Depth** in AI applications. To extract the hidden flags, players must bypass two layers of security:
1. **The AI's Persona (System Prompt):** Each level features a different Alice in Wonderland character (Mad Hatter, Cheshire Cat, Queen of Hearts) with increasing levels of paranoia.
2. **The Python Security Filter:** Even if the AI slips up, a custom Python filter inspects the output to block direct leaks, reversed text, Base64 encoding, and formatting tricks.

## 🏗️ Architecture
* **Frontend:** HTML, CSS (Glassmorphism UI), Vanilla JavaScript.
* **Backend:** Python (FastAPI).
* **AI Engine:** Ollama (running the `qwen3.5:4b` model locally).

> ⚠️ **IMPORTANT: Local Host Agent Required to Play**
> Because this game runs entirely locally to protect privacy and avoid cloud API costs, it **cannot** be played by just opening the HTML file. You must have the local Python backend and Ollama running on your machine to play.

## 🚀 How to Play (Local Setup)

### 1. Prerequisites
You need the following installed on your machine:
* [Python](https://www.python.org/downloads/)
* [Ollama](https://ollama.com/download)

### 2. Pull the AI Model
Open your terminal (PowerShell or Bash) and download the Qwen model:
```bash
ollama pull qwen3.5:4b
