import oolama
import gradio as gr
import torch
import hypernetx as hnx

def build_hypergraph(input_data):
    tokens = oolama.tokenize(input_data)
    entities = oolama.extract_entities(tokens)
    relations = oolama.extract_relations(tokens)

    hypergraph = hnx.Hypergraph()
    for entity in entities:
        hypergraph.add_node(entity)
    for relation in relations:
        hypergraph.add_edge(relation['entities'], relation['type'])

    return hypergraph

def infer_on_hypergraph(hypergraph):
    oolama_input = hnx.convert_to_oolama_format(hypergraph)
    results = oolama.infer(oolama_input)
    return results

def format_hypergraph_inference_results(results):
    formatted_results = oolama.format_results(results)
    return formatted_results

def process_data(input_data):
    hypergraph = build_hypergraph(input_data)
    results = infer_on_hypergraph(hypergraph)
    formatted_results = format_hypergraph_inference_results(results)
    return formatted_results

def gradio_interface(input_data):
    result = process_data(input_data)
    return result

iface = gr.Interface(
    fn=gradio_interface,
    inputs="text",
    outputs="text",
    title="Aprimorador de Dados com Hipergrafos",
    description="Utiliza oolama e hipergrafos para aprimorar os dados fornecidos."
)

if __name__ == "__main__":
    iface.launch()
