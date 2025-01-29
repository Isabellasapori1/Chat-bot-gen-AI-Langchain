from flask import Flask, request, jsonify, render_template
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os
import requests
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM

app = Flask(__name__)

# Configuração do VectorStore e Unsplash
persist_directory = "./chroma_db"
UNSPLASH_ACCESS_KEY = "bpY8aGhxsP3FPQ0UtCY_RyWUSY0Bs9TxGXWbnHKlOV4"

def load_vectorstore():
    try:
        vectorstore = Chroma(
            embedding_function=OllamaEmbeddings(model="llama3"),
            persist_directory=persist_directory,
        )
        return vectorstore
    except Exception as e:
        print(f"Erro ao carregar o VectorStore: {e}")
        return None

vectorstore = load_vectorstore()

def fetch_image_from_unsplash(query):
    """
    Busca uma imagem no Unsplash com base no termo de busca fornecido.
    """
    try:
        url = f"https://api.unsplash.com/photos/random?query={query}&client_id={UNSPLASH_ACCESS_KEY}"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        return {'image_url': data["urls"]["regular"]}
    except Exception as e:
        return {'error': f"Erro ao buscar imagem: {str(e)}"}

def generate_graph(chart_type, variables, values):
    """
    Gera um gráfico baseado no tipo, variáveis e valores fornecidos, usando Seaborn.
    """
    try:
        if not variables or not values:
            return {'error': 'Variáveis ou valores ausentes para o gráfico.'}

        data = pd.DataFrame({'Variáveis': variables, 'Valores': values})
        plt.figure(figsize=(6, 4))

        if chart_type == "bar":
            sns.barplot(x="Variáveis", y="Valores", data=data, palette="muted")
        elif chart_type == "line":
            sns.lineplot(x="Variáveis", y="Valores", data=data, marker='o')
        else:
            return {'error': 'Tipo de gráfico não suportado.'}

        plt.title('Gráfico Gerado')
        plt.xlabel('Variáveis')
        plt.ylabel('Valores')

        graph_path = "static/graph.png"
        plt.savefig(graph_path)
        plt.close()

        return {'graph_url': graph_path}
    except Exception as e:
        return {'error': str(e)}

def process_graph_request(user_message):
    """
    Processa mensagens para criar gráficos, extraindo variáveis e valores.
    """
    chart_type = "bar" if "barras" in user_message.lower() else "line"
    variables = []
    values = []

    try:
        if "variáveis" in user_message.lower():
            variables = user_message.split("variáveis")[1].split("valores")[0].strip().split(", ")
        if "valores" in user_message.lower():
            values = [float(x) for x in user_message.split("valores")[1].strip().split(", ")]
    except Exception as e:
        print(f"Erro ao extrair variáveis/valores: {e}")
        return {'bot_message': 'Não consegui entender as variáveis e valores para o gráfico. Tente algo como: "gráfico de barras com variáveis A, B, C e valores 10, 20, 30."'}

    result = generate_graph(chart_type, variables, values)
    if 'error' in result:
        return {'bot_message': result['error']}
    
    return {'bot_message': 'Aqui está o gráfico que você pediu!', 'graph_url': result['graph_url']}

@app.route('/')
def index():
    return render_template('tela.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message', '')
    if not user_message:
        return jsonify({'bot_message': 'Por favor, envie uma mensagem válida.'})

    try:
        # Verificar se a mensagem termina com "-livro"
        if user_message.endswith('-livro'):
            user_message = user_message.replace('-livro', '').strip()
            llm = OllamaLLM(model="llama3")

            docs = vectorstore.similarity_search(user_message, k=5) if vectorstore else []
            context = "\n\n".join([doc.page_content for doc in docs]) if docs else "Nenhum contexto encontrado sobre o livro."

            prompt = (
                f"Contexto:\n{context}\n\n"
                f"Pergunta: {user_message}\n"
                "Responda de forma clara e objetiva sobre o livro."
            )
            bot_message = llm.invoke(prompt).strip()
            return jsonify({'bot_message': bot_message})

        # Verificar se a mensagem termina com "-grafico"
        elif user_message.endswith('-grafico'):
            user_message = user_message.replace('-grafico', '').strip()
            response = process_graph_request(user_message)
            if 'graph_url' in response:
                return jsonify({'bot_message': response['bot_message'], 'graph_url': response['graph_url']})
            else:
                return jsonify({'bot_message': response['bot_message']})

        # Verificar se a mensagem termina com "-imagem"
        elif user_message.endswith('-imagem'):
            user_message = user_message.replace('-imagem', '').strip()
            response = fetch_image_from_unsplash(user_message)
            if 'image_url' in response:
                return jsonify({'bot_message': f"Aqui está a imagem relacionada ao tema '{user_message}':", 'image_url': response['image_url']})
            else:
                return jsonify({'bot_message': response['error']})

        # Resposta padrão para mensagens gerais
        else:
            llm = OllamaLLM(model="llama3")
            prompt = (
                f"Pergunta: {user_message}\n"
                "Responda de forma amigável, curta e direta."
            )
            bot_message = llm.invoke(prompt).strip()
            return jsonify({'bot_message': bot_message})
    except Exception as e:
        return jsonify({'bot_message': f"Erro ao processar: {str(e)}"})

if __name__ == '__main__':
    os.makedirs('static', exist_ok=True)  # Garantir que o diretório estático exista
    app.run(debug=True)
