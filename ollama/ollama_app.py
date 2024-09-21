"""
This module provides functionalities for processing input data and generating hypergraphs.
"""

import streamlit as st
import ollama
import os
import json
import time
import hypernetx as hnx
from transformers import AutoTokenizer, AutoModel

class Node:
    """
    Represents a node in the hypergraph.
    """
    def __init__(self, node_id, title, content, embedding=None):
        self.id = node_id
        self.title = title
        self.content = content
        self.embedding = embedding

class Hypergraph:
    """
    Represents a hypergraph with nodes and edges.
    """
    def __init__(self):
        self.nodes = {}
        self.edges = []

def process_input_data(input_data):
    """
    Process the input data to extract entities and relations, and create a hypergraph.

    Args:
        input_data (str): The input data to process.

    Returns:
        hnx.Hypergraph: The generated hypergraph.
    """
    try:
        tokens = ollama.tokenize(input_data)
        entities = ollama.extract_entities(tokens)
        relations = ollama.extract_relations(tokens)

        hypergraph = hnx.Hypergraph()
        for entity in entities:
            hypergraph.add_node(entity)
        for relation in relations:
            hypergraph.add_edge(relation['entities'], relation['type'])

        return hypergraph
    except ollama.OllamaError as e:
        st.error(f"Error processing input data: {e}")
        return None

def infer_on_hypergraph(hypergraph):
    """
    Perform inference on the given hypergraph.

    Args:
        hypergraph (hnx.Hypergraph): The hypergraph to perform inference on.

    Returns:
        dict: The inference results.
    """
    try:
        ollama_input = hnx.convert_to_ollama_format(hypergraph)
        results = ollama.infer(ollama_input)
        return results
    except ollama.OllamaError as e:
        st.error(f"Error during inference: {e}")
        return None

def format_hypergraph_inference_results(results):
    """
    Format the inference results for display.

    Args:
        results (dict): The inference results to format.

    Returns:
        str: The formatted results.
    """
    try:
        formatted_results = "\n".join([f"{key}: {value}" for key, value in results.items()])
        return formatted_results
    except Exception as e:
        st.error(f"Error formatting results: {e}")
        return ""

def main():
    """
    Main function to run the Streamlit app.
    """
    st.title("Hypergraph Inference App")

    input_data = st.text_area("Enter input data:")
    if st.button("Process"):
        hypergraph = process_input_data(input_data)
        if hypergraph:
            results = infer_on_hypergraph(hypergraph)
            if results:
                formatted_results = format_hypergraph_inference_results(results)
                st.text_area("Inference Results:", formatted_results)

if __name__ == "__main__":
    main()