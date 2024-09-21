import streamlit as st
import groq
import os
import json
import time
import hypernetx as hnx  # Importando biblioteca para manipular hipergrafos [[4]](https://poe.com/citation?message_id=253875365866&citation=4)

client = groq.Groq()

# Função para construir o hipergrafo a partir do prompt
def build_hypergraph(input_data):
    # Aqui, tokenizamos o input e extraímos entidades e relações
    tokens = oolama.tokenize(input_data)  # Tokenização com Oolama [[4]](https://poe.com/citation?message_id=253875365866&citation=4)
    entities = oolama.extract_entities(tokens)  # Extração de entidades
    relations = oolama.extract_relations(tokens)  # Extração de relações

    # Criamos o hipergrafo utilizando hypernetx [[4]](https://poe.com/citation?message_id=253875365866&citation=4)
    hypergraph = hnx.Hypergraph()
    
    # Adicionamos as entidades como nós
    for entity in entities:
        hypergraph.add_node(entity)
    
    # Adicionamos as relações como hiperarestas
    for relation in relations:
        hypergraph.add_edge(relation['entities'], relation['type'])  # Adiciona hiperarestas ao hipergrafo [[6]](https://poe.com/citation?message_id=253875365866&citation=6)
    
    return hypergraph

# Função para realizar inferência no hipergrafo
def infer_on_hypergraph(hypergraph):
    # Convertendo o hipergrafo para o formato apropriado para inferência com Oolama [[6]](https://poe.com/citation?message_id=253875365866&citation=6)
    oolama_input = hnx.convert_to_oolama_format(hypergraph)  # Conversão para formato compatível com Oolama
    results = oolama.infer(oolama_input)  # Realizando a inferência

    return results

# Função para formatar os resultados
def format_hypergraph_inference_results(results):
    formatted_results = oolama.format_results(results)  # Formatando os resultados da inferência [[3]](https://poe.com/citation?message_id=253875365866&citation=3)
    return formatted_results

# Função que gera a resposta baseada no hipergrafo
def generate_hypergraph_response(prompt):
    # Construção inicial do hipergrafo a partir do prompt
    hypergraph = build_hypergraph(prompt)
    
    # Realiza inferências no hipergrafo
    results = infer_on_hypergraph(hypergraph)
    
    # Formata os resultados para serem exibidos
    formatted_results = format_hypergraph_inference_results(results)
    
    return formatted_results

def main():
    st.set_page_config(page_title="g1 prototype", page_icon="🧠", layout="wide")
    
    st.title("g1: Usando Llama-3.1 70b no Groq com Cadeias de Raciocínio Baseadas em Arquitetura de Hipergrafos")
    
    st.markdown("""
    Este é um protótipo inicial que utiliza métodos heurísticos e hermenêuticos com uma arquitetura de hipergrafos para melhorar a precisão das saídas. 
    """)
    
    # Caixa de texto para a consulta do usuário
    user_query = st.text_input("Insira sua consulta:", placeholder="Por exemplo, Quantos 'R's existem na palavra morango?")
    
    if user_query:
        st.write("Gerando resposta com base na arquitetura de hipergrafos...")
        
        # Geramos a resposta com base no hipergrafo
        response = generate_hypergraph_response(user_query)
        
        # Exibimos a resposta formatada
        st.write(response)

if __name__ == "__main__":
    main()
