import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import ollama

# Load the local LLaMA model using ollama
tokenizer = AutoTokenizer.from_pretrained("path_to_local_llama_model")
model = AutoModelForCausalLM.from_pretrained("path_to_local_llama_model")

def generate_text(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=50)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text

iface = gr.Interface(
    fn=generate_text,
    inputs=gr.Textbox(lines=2, placeholder="Digite seu prompt aqui..."),
    outputs="text",
    title="Gerador de Texto com LLaMA",
)

iface.launch()