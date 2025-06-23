# 🧠 Desktop Automation Bot

An intelligent desktop assistant that performs real-time automation on Windows using **voice commands** and **visual understanding of the screen** — powered by Python, computer vision, and LLMs. No hardcoded scripts. Just intelligent task execution.

---

## 🧠 Core Idea

You speak a high-level command like:

> “Check my IoT assignment”

The assistant:

* Understands the command using an LLM (e.g., LLaMA 3.3 hosted via Groq)
* Analyzes the screen visually
* Matches the intent to visible elements
* Clicks, types, and automates step-by-step until the task is complete

---

## 🎤 Voice-Based Interaction

* **Voice Input**: Captured via your microphone using `speech_recognition`
* **Voice Output**: Bot replies and confirms actions using `pyttsx3` (offline text-to-speech)

---

## 🔄 How It Works

1. **🎙 Voice Input:**

   * Your voice is captured using the `speech_recognition` library and converted to text.

2. **💡 Command Understanding:**

   * The transcribed command is sent to an LLM (via Groq API).
   * It returns a list of subtasks such as:

     * “Open Chrome”
     * “Go to Google Classroom”
     * “Click on IoT assignment”

3. **🖥️ Visual Detection:**

   * A screenshot of the desktop is taken using PyAutoGUI or MSS.
   * YOLO detects all clickable UI elements (icons, buttons, etc.).
   * Each detected item is passed to Florence (770M parameter model) to generate a descriptive caption.

4. **🔗 Matching Tasks to UI:**

   * The LLM receives the list of subtasks along with the detected UI items (descriptions + IDs).
   * It selects the most relevant clickable element for each subtask.

5. **⚙️ UI Automation:**

   * PyAutoGUI simulates mouse clicks, keystrokes, or scroll actions.
   * The assistant loops back to the desktop and moves to the next subtask.

---

## ⚙️ Technologies Used

| Component             | Tool/Library           |
| --------------------- | ---------------------- |
| Voice Input           | `speech_recognition`   |
| Text-to-Speech Output | `pyttsx3`              |
| Screen Capture        | `PyAutoGUI`            |
| Object Detection      | `YOLO`                 |
| Image Captioning      | `Florence` (770M)      |
| Language Model (LLM)  | `LLaMA` (via Groq API) |
| Automation Execution  | `PyAutoGUI`            |
| Icon Caching\Retrival | `FAISS`  `RESNET`      |
| Language              | Python                 |

---

## 🛠️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/MuhammadAhmadBajwa/Desktop-Automation.git
cd Desktop-Automation
```

### 2. Create and Activate Virtual Environment

```bash
python -m venv venv
```

Activate it:

* **Windows**:

  ```bash
  venv\Scripts\activate
  ```
* **macOS/Linux**:

  ```bash
  source venv/bin/activate
  ```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Assistant

```bash
python app.py
```

---


---

## ⚠️ Challenges

* ⏳ **Captioning is slow** on CPU (\~10s/icon) when encountering new UI elements
* 🔲 **YOLO may produce inaccurate bounding boxes** in cluttered or dense UI layouts
* 🧾 **Florence model can miscaption rare or ambiguous UI elements**
* 📸 **No existing dataset** for UI component captioning — requires manual labeling for training
* 🧠 **Solution:** Added a **vector search–based caching mechanism** using **ResNet image embeddings**
  → As the assistant is used more often, most common icons/UI elements are **cached**, reducing captioning time dramatically on repeated use
---


---

## 🎯 Project Goal

To build a **real-time, intelligent automation system** that adapts to desktop environments, understands **natural language**, and performs tasks without needing static scripts.



