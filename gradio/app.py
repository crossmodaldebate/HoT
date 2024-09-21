import oolama
import gradio as gr
import hypernetx as hnx
import groq
import time
import json

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

def format_results(results):
    formatted = oolama.format_results(results)
    return formatted

def make_api_call(client, messages, max_tokens, is_final_answer=False):
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model="llama-3.1-70b-versatile",
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.2,
                response_format={"type": "json_object"}
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            if attempt == 2:
                if is_final_answer:
                    return {"title": "Erro", "content": f"Falha ao gerar a resposta final após 3 tentativas. Erro: {str(e)}"}
                else:
                    return {"title": "Erro", "content": f"Falha ao gerar a etapa após 3 tentativas. Erro: {str(e)}", "next_action": "resposta_final"}
            time.sleep(1)

def generate_response(client, prompt):
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
        step_data = make_api_call(client, messages, 300)
        end_time = time.time()
        thinking_time = end_time - start_time
        total_thinking_time += thinking_time
        
        if step_data.get('title') == "Erro":
            steps.append((f"Etapa {step_count}: {step_data.get('title')}", step_data.get('content'), thinking_time))
            break
        
        step_title = f"Etapa {step_count}: {step_data.get('title', 'Sem Título')}"
        step_content = step_data.get('content', 'Sem Conteúdo')
        steps.append((step_title, step_content, thinking_time))
        
        messages.append({"role": "assistant", "content": json.dumps(step_data)})
        
        if step_data.get('next_action') == 'resposta_final':
            break
        
        step_count += 1

    messages.append({"role": "user", "content": "Por favor, forneça a resposta final com base no seu raciocínio acima."})
    
    start_time = time.time()
    final_data = make_api_call(client, messages, 200, is_final_answer=True)
    end_time = time.time()
    thinking_time = end_time - start_time
    total_thinking_time += thinking_time
    
    steps.append(("Resposta Final", final_data.get('content', 'Sem Conteúdo'), thinking_time))
    
    return steps, total_thinking_time

def format_steps(steps, total_time):
    html_content = ""
    for title, content, thinking_time in steps:
        if title == "Resposta Final":
            html_content += "<h3>{}</h3>".format(title)
            html_content += "<p>{}</p>".format(content.replace('\n', '<br>'))
        else:
            html_content += """
            <details>
                <summary><strong>{}</strong></summary>
                <p>{}</p>
                <p><em>Tempo de reflexão para esta etapa: {:.2f} segundos</em></p>
            </details>
            <br>
            """.format(title, content.replace('\n', '<br>'), thinking_time)
    html_content += "<strong>Tempo total de reflexão: {:.2f} segundos</strong>".format(total_time)
    return html_content

def main(api_key, user_query):
    if not api_key:
        return "Por favor, insira sua chave API do Groq para prosseguir.", ""
    
    if not user_query:
        return "Por favor, insira uma consulta para começar.", ""
    
    try:
        client = groq.Groq(api_key=api_key)
    except Exception as e:
        return f"Falha ao inicializar o cliente Groq. Erro: {str(e)}", ""
    
    try:
        steps, total_time = generate_response(client, user_query)
        formatted_steps = format_steps(steps, total_time)
    except Exception as e:
        return f"Ocorreu um erro durante o processamento. Erro: {str(e)}", ""
    
    return formatted_steps, ""

with gr.Blocks() as demo:
    gr.Markdown("# 🧠 g1: Usando Llama-3.1 70b no Groq para Criar Cadeias de Raciocínio Semelhantes ao O1")
    
    gr.Markdown("""
    Este é um protótipo inicial de uso de métodos heurísticos e hermenêuticos com uma arquitetura em hipergrafo para melhorar a precisão de saída. Não é perfeito e a precisão ainda não foi formalmente avaliada. 

    Repositório open source [aqui](https://github.com/bklieger-groq)
    """)
    
    with gr.Row():
        with gr.Column():
            api_input = gr.Textbox(
                label="Digite sua chave API do Groq:",
                placeholder="Sua chave API do Groq",
                type="password"
            )
            user_input = gr.Textbox(
                label="Digite sua consulta:",
                placeholder="ex: Quantos 'R's há na palavra morango?",
                lines=2
            )
            submit_btn = gr.Button("Gerar Resposta")
    
    with gr.Row():
        with gr.Column():
            output_html = gr.HTML()
    
    submit_btn.click(fn=main, inputs=[api_input, user_input], outputs=output_html)

if __name__ == "__main__":
    demo.launch()
