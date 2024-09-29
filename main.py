import gradio as gr
import whisper
from moviepy.editor import VideoFileClip
from pydub import AudioSegment
from shazamio import Shazam
import tempfile
import os
import asyncio
import torch
from pyannote.audio import Pipeline
import cv2
import numpy as np
import sqlite3
import threading
import time
import subprocess
from langchain_community.llms import Ollama
whisper_model = whisper.load_model("large")
OLLAMA = r"C:\Users\Nimtey\.ollama\ollama.exe"
def start_ollama_service():
    subprocess.run(OLLAMA + " pull llama3.2", shell=True)
OLLAMA_SERVICE_THREAD = threading.Thread(target=start_ollama_service)
OLLAMA_SERVICE_THREAD.start()
time.sleep(10)
llm = Ollama(base_url='http://127.0.0.1:11434', model="llama3.2")

def check_cuda():
    return torch.cuda.is_available()

def transcribe_speech(audio_file, language='ru'):
    result = whisper_model.transcribe(audio_file, language=language, word_timestamps=True)
    return result['text'], result['segments']

def extract_audio_from_video(video_file):
    audio_file = "extracted_audio.wav"
    with VideoFileClip(video_file) as video:
        audio = video.audio
        audio.write_audiofile(audio_file, codec='pcm_s16le')
    return audio_file

async def recognize_music(audio_segment):
    shazam = Shazam()
    
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio_file:
        temp_filename = temp_audio_file.name
        audio_segment.export(temp_filename, format='wav')
    
    out = await shazam.recognize(temp_filename)
    os.remove(temp_filename)
    
    return out

async def recognize_from_file(file):
    audio_file = extract_audio_from_video(file)
    audio = AudioSegment.from_file(audio_file)

    segment_length = 60 * 1000
    results = set()

    for start in range(0, len(audio), segment_length):
        segment = audio[start:start + segment_length]
        result = await recognize_music(segment)
        
        if 'track' in result:
            track = result['track']
            title = track.get('title', 'Неизвестно')
            performer = ', '.join(track.get('subtitle', 'Неизвестен'))
            results.add((title, performer))
        else:
            results.add(('Неизвестно', 'Неизвестен'))

    return results

def format_time(seconds):
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes}:{seconds:02d}"

def diarize_speakers(audio_file, segments):
    token = "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"  # Your token
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=token)
    device = torch.device('cuda' if check_cuda() else 'cpu')
    pipeline.to(device)
    
    diarization = pipeline({'uri': 'audio', 'audio': audio_file})
    
    diarization_result = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        speaker_words = []
        for segment in segments:
            for word in segment['words']:
                if word['start'] >= turn.start and word['end'] <= turn.end:
                    speaker_words.append(word['word'])
        
        start_time = format_time(turn.start)
        end_time = format_time(turn.end)

        diarization_result.append({
            'speaker': speaker,
            'words': " ".join(speaker_words),
            'start': start_time,
            'end': end_time
        })
    
    return diarization_result

def compare_images(frame, img_path):
    image = cv2.imread(img_path)
    if image is None:
        return 0

    if image.shape[0] > frame.shape[0] or image.shape[1] > frame.shape[1]:
        scale_factor = min(frame.shape[0] / image.shape[0], frame.shape[1] / image.shape[1])
        new_size = (int(image.shape[1] * scale_factor), int(image.shape[0] * scale_factor))
        image = cv2.resize(image, new_size)

    res = cv2.matchTemplate(frame, image, cv2.TM_CCOEFF_NORMED)
    return np.max(res)

def find_similar_images(video_file):
    current_directory = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(current_directory, 'Simvoly', 'osnov.db')
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT group_name, file_path FROM images")
    images = cursor.fetchall()

    video = cv2.VideoCapture(video_file)
    results = []

    while True:
        ret, frame = video.read()
        if not ret:
            break

        current_time = int(video.get(cv2.CAP_PROP_POS_MSEC)) // 1000

        for group_name, file_path in images:
            similarity = compare_images(frame, file_path)
            if similarity > 0.7:
                results.append(f"Найдено изображение в группе: {group_name} на {current_time} секунд(ы) с вероятностью {similarity:.2f}")

        video.set(cv2.CAP_PROP_POS_MSEC, (current_time // 5 + 1) * 5000)

    video.release()
    conn.close()

    return results

def generate_tags(text):
    prompt = f"Напиши теги на основе текста: {text}"
    response = llm.invoke(prompt)
    return response

def process_file(file):
    if file is None:
        return "Файл не загружен.", "", "", "", ""

    file_name = file.name if hasattr(file, 'name') else file
    file_path = file.name if hasattr(file, 'name') else str(file)

    if file_name.endswith(('.mp4', '.mkv', '.avi')):
        audio_file = extract_audio_from_video(file_path)
        transcription, segments = transcribe_speech(audio_file)
        music_results = asyncio.run(recognize_from_file(file_path))
        
        diarization_results = diarize_speakers(audio_file, segments)
        image_results = find_similar_images(file_path)

        diarization_str = "\n".join([f"Спикер: {d['speaker']}, Слова: {d['words']} (с {d['start']} до {d['end']})" for d in diarization_results])
        music_results_str = "\n".join([f"{title} - {performer}" for title, performer in music_results])
        tags = generate_tags(transcription)

        return transcription, music_results_str, diarization_str, "\n".join(image_results), tags
    elif file_name.endswith(('.wav', '.mp3', '.ogg')):
        transcription, _ = transcribe_speech(file_path)
        tags = generate_tags(transcription)
        return transcription, "Музыка не распознаётся из аудио файлов.", "", "", tags
    else:
        return "Неподдерживаемый формат файла.", "", "", "", ""

# Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("## Транскрибатор и Распознавание Музыки")

    file_input = gr.File(label="Загрузите аудио или видео файл")
    text_output = gr.Textbox(label="Распознанный текст", lines=4)
    music_output = gr.Textbox(label="Распознанная музыка", lines=4)
    diarization_output = gr.Textbox(label="Диаризация спикеров", lines=4)
    image_output = gr.Textbox(label="Сравнение изображений", lines=4)
    tags_output = gr.Textbox(label="Сгенерированные теги", lines=4)

    file_input.change(fn=process_file, inputs=file_input, outputs=[text_output, music_output, diarization_output, image_output, tags_output])
demo.launch(share=True)
