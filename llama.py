import os
import subprocess
from langchain_community.llms import Ollama

class TagGenerator:
    def __init__(self, ollama_path=r"C:\Users\Nimtey\.ollama\ollama.exe"):
        if not os.path.exists(ollama_path):
            subprocess.run(f"curl -L https://ollama.com/download/ollama-windows-amd64 -o {ollama_path}", shell=True)
        
        subprocess.run(f"{ollama_path} pull llama3.2", shell=True)
        
        self.llm = Ollama(base_url='http://127.0.0.1:11434', model="llama3.2")

    def generate_tags(self, text):
        prompt = f"Напиши теги на основе текста: {text}"
        response = self.llm.invoke(prompt)
        return response
if __name__ == "__main__":
    generator = TagGenerator()
    text = "Психология — это наука, изучающая поведение и психические процессы человека."
    tags = generator.generate_tags(text)
    print(tags)
