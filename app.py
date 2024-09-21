import streamlit as st
import groq
import hypernetx as hnx
import oolama  # Certifique-se de que o módulo oolama esteja disponível

client = groq.Groq()

def build_hypergraph(input_data):
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
    except Exception as e:
        st.error(f"Erro ao construir o hipergrafo: {e}")
        return None

def infer_on_hypergraph(hypergraph):
    try:
        oolama_input = hnx.convert_to_oolama_format(hypergraph)
        results = oolama.infer(oolama_input)
        return results
    except Exception as e:
        st.error(f"Erro ao realizar inferência no hipergrafo: {e}")
        return None

def format_hypergraph_inference_results(results):
    try:
        formatted_results = oolama.format_results(results)
        return formatted_results
    except Exception as e:
        st.error(f"Erro ao formatar os resultados da inferência: {e}")
        return None

def generate_hypergraph_response(prompt):
    hypergraph = build_hypergraph(prompt)
    if hypergraph is None:
        return "Erro ao gerar hipergrafo."
    
    results = infer_on_hypergraph(hypergraph)
    if results is None:
        return "Erro ao realizar inferência no hipergrafo."
    
    formatted_results = format_hypergraph_inference_results(results)
    if formatted_results is None:
        return "Erro ao formatar os resultados da inferência."
    
    return formatted_results

def main():
    st.set_page_config(page_title="g1 prototype", page_icon="🧠", layout="wide")
    
    st.title("g1: Usando Llama-3.1 70b no Groq com Cadeias de Raciocínio Baseadas em Arquitetura de Hipergrafos")
    
    st.markdown("""
    Este é um protótipo inicial que utiliza métodos heurísticos e hermenêuticos com uma arquitetura de hipergrafos para melhorar a precisão das saídas. 
    """)
    
    user_query = st.text_input("Insira sua consulta:", placeholder="Por exemplo, Quantos 'R's existem na palavra morango?")
    
    if user_query:
        st.write("Gerando resposta com base na arquitetura de hipergrafos...")
        
        response = generate_hypergraph_response(user_query)
        
        st.write(response)

if _name_ == "_main_":
    main()
