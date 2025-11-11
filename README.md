---
title: Haarcascade Face Detection
emoji: 😎
colorFrom: indigo
colorTo: pink
sdk: gradio
sdk_version: "4.44.1"
app_file: app.py
pinned: false
---




# 🧠 Real-time Face Detection using OpenCV & Gradio

This project is a **real-time face detection web app** built with **Gradio** and **OpenCV's Haarcascade classifier**.  
It allows users to **stream their webcam feed directly from the browser** and see live face detection results — no local setup required!

---

## 🚀 Features
- 🎥 **Real-time webcam streaming** via Gradio interface  
- 🧩 **Face detection** using OpenCV’s `haarcascade_frontalface_default.xml`  
- ⚙️ Adjustable **Scale Factor** and **Min Neighbors** sliders to fine-tune detection sensitivity  
- 💻 Works seamlessly on **Hugging Face Spaces** or **locally**  
- 🪶 Lightweight — no deep learning model required  

---

## 🧩 Tech Stack
- **Gradio** — Interactive web UI  
- **OpenCV (Haarcascade)** — Classical face detection  
- **Pillow** — Image conversion  
- **NumPy** — Image array manipulation  

---

## 📦 Requirements
Make sure your `requirements.txt` contains:

