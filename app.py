import gradio as gr
import networkx as nx
import matplotlib.pyplot as plt
from ollama import Ollama

# Initialize the LLaMA model
ollama = Ollama(model_path="path/to/llama-3.1.GGUF")

def generate_text(prompt):
    response = ollama.generate(prompt, max_tokens=50)
    return response['text']

def create_directional_graph():
    # Create a directed graph
    G = nx.DiGraph()

    # Add nodes
    G.add_node(1)
    G.add_node(2)
    G.add_node(3)
    G.add_node(4)

    # Add edges
    G.add_edge(1, 2)
    G.add_edge(2, 3)
    G.add_edge(3, 4)
    G.add_edge(4, 1)

    return G

def visualize_graph(G):
    # Draw the graph
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=2000, edge_color='gray', arrows=True)
    plt.title('Directional Graph')
    plt.show()

def main(prompt):
    # Generate text
    generated_text = generate_text(prompt)
    
    # Create and visualize the directional graph
    G = create_directional_graph()
    visualize_graph(G)
    
    return generated_text

iface = gr.Interface(
    fn=main,
    inputs=gr.Textbox(lines=2, placeholder="Digite seu prompt aqui..."),
    outputs="text",
    title="Gerador de Texto com LLaMA e Visualização de Grafo Direcional",
)

if __name__ == "__main__":
    iface.launch()