"""
This module provides functionalities for processing input data and generating hypergraphs.
"""

import streamlit as st
import groq
import hypernetx as hnx
import oolama

def process_input_data(input_data):
    """
    Process the input data to extract entities and relations, and create a hypergraph.

    Args:
        input_data (str): The input data to process.

    Returns:
        hnx.Hypergraph: The generated hypergraph.
    """
    try:
        tokens = oolama.tokenize(input_data)
        entities = oolama.extract_entities(tokens)
        relations = oolama.extract_relations(tokens)

        hypergraph = hnx.Hypergraph()
        for entity in entities:
            hypergraph.add_node(entity)
        for relation in relations:
            hypergraph.add_edge(relation['entities'], relation['type'])

        return hypergraph
    except oolama.OolamaError as e:
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
        oolama_input = hnx.convert_to_oolama_format(hypergraph)
        results = oolama.infer(oolama_input)
        return results
    except oolama.OolamaError as e:
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