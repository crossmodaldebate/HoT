import streamlit as st
import ollama
import os
import json
import time

def make_api_call(messages, max_tokens, is_final_answer=False):
    for attempt in range(3):
        try:
            response = ollama.chat(
                model="llama3.1:70b",
                messages=messages,
                options={"temperature":0.2, "max_length":max_tokens},
                format='json',
            )
            return json.loads(response['message']['content'])
        except Exception as e:
            if attempt == 2:
                if is_final_answer:
                    return {"title": "Erro", "content": f"Falha ao gerar a resposta final após 3 tentativas. Erro: {str(e)}"}
                else:
                    return {"title": "Erro", "content": f"Falha ao gerar etapa após 3 tentativas. Erro: {str(e)}", "next_action": "resposta_final"}
            time.sleep(1)  # Espera por 1 segundo antes de tentar novamente

def generate_response(prompt):
    messages = [
        {"role": "system", "content": """Você é um assistente de IA especialista que utiliza métodos heurísticos e hermenêuticos com uma arquitetura em hipergrafo. Para cada etapa, forneça um título que descreva o objetivo da etapa e o conteúdo explicativo. Siga a estrutura de um hipergrafo para decompor o problema em nós e suas relações.

Exemplo de resposta JSON válida:
json
{
    "title": "Identificação de Nós Principais",
    "content": "Para começar a resolver este problema, precisamos identificar os nós principais e suas relações dentro do hipergrafo. Isso envolve...",
    "next_action": "continuar"
}
"""},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": "Obrigado! Vou começar a pensar passo a passo seguindo minhas instruções, iniciando pela decomposição do problema em uma arquitetura de hipergrafo."}
    ]
    
    steps = []
    step_count = 1
    total_thinking_time = 0
    
    while True:
        start_time = time.time()
        step_data = make_api_call(messages, 300)
        end_time = time.time()
        thinking_time = end_time - start_time
        total_thinking_time += thinking_time
        
        steps.append((f"Etapa {step_count}: {step_data['title']}", step_data['content'], thinking_time))
        
        messages.append({"role": "assistant", "content": json.dumps(step_data)})
        
        if step_data['next_action'] == 'resposta_final' or step_count > 25:  # Máximo de 25 etapas para evitar tempo de pensamento infinito. Pode ser ajustado.
            break
        
        step_count += 1

        # Yield após cada etapa para o Streamlit atualizar
        yield steps, None  # Não estamos retornando o tempo total até o final

    # Gerar resposta final
    messages.append({"role": "user", "content": "Por favor, forneça a resposta final com base no seu raciocínio acima."})
    
    start_time = time.time()
    final_data = make_api_call(messages, 200, is_final_answer=True)
    end_time = time.time()
    thinking_time = end_time - start_time
    total_thinking_time += thinking_time
    
    steps.append(("Resposta Final", final_data['content'], thinking_time))

    yield steps, total_thinking_time

def main():
    st.set_page_config(page_title="g1 protótipo", page_icon="🧠", layout="wide")
    
    st.title("g1: Usando Llama-3.1 70b no Ollama para criar cadeias de raciocínio semelhantes ao o1 com Arquitetura de Hipergrafo")
    
    st.markdown("""
    Este é um protótipo inicial de uso de métodos heurísticos e hermenêuticos com uma arquitetura em hipergrafo para melhorar a precisão de saída. Não é perfeito e a precisão ainda não foi formalmente avaliada. 

    Repositório open source [aqui](https://github.com/bklieger-groq)
    """)
    
    # Entrada de texto para consulta do usuário
    user_query = st.text_input("Digite sua consulta:", placeholder="ex: Quantos 'R's há na palavra morango?")
    
    if user_query:
        st.write("Gerando resposta...")
        
        # Criar elementos vazios para armazenar o texto gerado e o tempo total
        response_container = st.empty()
        time_container = st.empty()
        
        # Gerar e exibir a resposta
        for steps, total_thinking_time in generate_response(user_query):
            with response_container.container():
                for i, (title, content, thinking_time) in enumerate(steps):
                    if title.startswith("Resposta Final"):
                        st.markdown(f"### {title}")
                        st.markdown(content.replace('\n', '<br>'), unsafe_allow_html=True)
                    else:
                        with st.expander(title, expanded=True):
                            st.markdown(content.replace('\n', '<br>'), unsafe_allow_html=True)
            
            # Mostrar o tempo total apenas quando estiver disponível no final
            if total_thinking_time is not None:
                time_container.markdown(f"Tempo total de pensamento: {total_thinking_time:.2f} segundos")

if _name_ == "_main_":
    main()
