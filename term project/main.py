import tkinter as tk
from tkinter import font
import sounddevice as sd
import numpy as np
import librosa
import threading
import time
from collections import deque

SAMPLE_RATE = 44100
BLOCK_DURATION = 0.3
MAX_HISTORY = 150

CHORDS = {
    "C": {"C", "E", "G"},
    "G": {"G", "B", "D"},
    "D": {"D", "F#", "A"},
    "Em": {"E", "G", "B"},
    "Am": {"A", "C", "E"},
}

running = False
score_history = deque(maxlen=MAX_HISTORY)

def freq_to_note(freq):
    if freq <= 0:
        return None
    note_num = int(round(12 * np.log2(freq / 440.0) + 69))
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F',
                  'F#', 'G', 'G#', 'A', 'A#', 'B']
    return note_names[note_num % 12]

def detect_notes(audio):
    pitches, magnitudes = librosa.piptrack(y=audio, sr=SAMPLE_RATE)
    notes = set()
    threshold = 0.01
    for i in range(pitches.shape[1]):
        index = magnitudes[:, i].argmax()
        freq = pitches[index, i]
        mag = magnitudes[index, i]
        if mag < threshold:
            continue
        note = freq_to_note(freq)
        if note:
            notes.add(note)
    return notes

def detect_score(detected_notes, target_chord):
    if not detected_notes:
        return 0.0
    matched_notes = detected_notes & target_chord
    score = len(matched_notes) / len(target_chord)
    return score

# GUI
def draw_gauge(canvas, score):
    canvas.delete("all")
    x, y, r = 120, 120, 90
    start_angle = -90
    extent = score * 360
    color = "#00ff00" if score >= 0.9 else "#ffa500" if score >= 0.7 else "#ff0000"
    canvas.create_oval(x-r, y-r, x+r, y+r, outline="#555555", width=15)
    canvas.create_arc(x-r, y-r, x+r, y+r, start=start_angle, extent=extent, style="arc", outline=color, width=15)
    canvas.create_text(x, y, text=f"{int(score*100)}%", fill="white", font=("Arial", 22, "bold"))

def draw_graph(canvas):
    canvas.delete("all")
    width, height = 700, 150
    for i, s in enumerate(score_history):
        x0 = i * (width / MAX_HISTORY)
        y0 = height
        y1 = height - s*height
        color = "#00ff00" if s >= 0.9 else "#ffa500" if s >= 0.7 else "#ff5555"
        canvas.create_line(x0, y0, x0, y1, fill=color, width=2)


# 실시간 오디오 루프
def audio_loop(target_chord_name, gauge_canvas, graph_canvas):
    global running
    target_chord = CHORDS.get(target_chord_name.get(), {"C","E","G"})
    while running:
        audio_chunk = sd.rec(int(SAMPLE_RATE*BLOCK_DURATION), samplerate=SAMPLE_RATE, channels=1, blocking=True).flatten()
        notes = detect_notes(audio_chunk)
        score = detect_score(notes, target_chord)
        score_history.append(score)

        draw_gauge(gauge_canvas, score)
        draw_graph(graph_canvas)
        time.sleep(0.05)

# GUI
def start():
    global running
    if not running:
        running = True
        threading.Thread(target=audio_loop, args=(target_chord_name, gauge_canvas, graph_canvas), daemon=True).start()

def stop():
    global running
    running = False

root = tk.Tk()
root.title("기타 연습 프로그램!")
root.configure(bg="#222222")
root.geometry("800x500")

title_font = font.Font(size=22, weight="bold")
tk.Label(root, text="실시간 기타 코드 정확도", font=title_font, fg="white", bg="#222222").pack(pady=10)

# 연습 코드 입력
frame_input = tk.Frame(root, bg="#222222")
tk.Label(frame_input, text="연습할 코드:", fg="white", bg="#222222").pack(side="left")
target_chord_name = tk.StringVar(value="C")
tk.Entry(frame_input, textvariable=target_chord_name, width=5, font=("Arial",14)).pack(side="left", padx=5)
frame_input.pack(pady=5)

# 버튼
button_frame = tk.Frame(root, bg="#222222")
tk.Button(button_frame, text="Start", width=12, command=start, bg="#00ffcc", fg="black", font=("Arial",12,"bold")).pack(side="left", padx=5)
tk.Button(button_frame, text="Stop", width=12, command=stop, bg="#ff5555", fg="white", font=("Arial",12,"bold")).pack(side="left", padx=5)
button_frame.pack(pady=5)

# 원형 게이지
gauge_canvas = tk.Canvas(root, width=250, height=250, bg="#222222", highlightthickness=0)
gauge_canvas.pack(pady=10)

# 단순 그래프
graph_canvas = tk.Canvas(root, width=700, height=150, bg="#111111", highlightthickness=0)
graph_canvas.pack(pady=10)

root.mainloop()
