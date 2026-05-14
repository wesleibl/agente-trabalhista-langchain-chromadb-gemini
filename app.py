import os
import re
import streamlit as st
from langchain_community.vectorstores import Chroma
from google import genai
from dotenv import load_dotenv

load_dotenv()

PERSIST_DIRECTORY = "./chroma_rh"
EMBEDDING_MODEL   = "models/gemini-embedding-001"
LLM_MODEL         = "gemini-2.5-flash-lite"
BASE_DIR          = os.path.dirname(os.path.abspath(__file__))

def coletar_contexto_usuario():
    with st.sidebar:
        st.header("📋 Seu perfil")

        polo = st.radio(
            "Você é:",
            ["Empregado / Trabalhador", "Empregador / Empresa / RH"],
            key="polo"
        )

        tipo_vinculo = st.selectbox(
            "Tipo de vínculo:",
            ["CLT Geral", "Doméstico", "Estagiário", "Terceirizado / Temporário", "Autônomo / PJ"],
            key="vinculo"
        )

        confirmado = st.button("✅ Confirmar perfil")

    mapa_vinculo = {
        "CLT Geral":                 "clt_geral",
        "Doméstico":                 "domestico",
        "Estagiário":                "estagiario",
        "Terceirizado / Temporário": "terceirizado",
        "Autônomo / PJ":             "geral",
    }

    if confirmado or st.session_state.get("perfil_confirmado"):
        st.session_state["perfil_confirmado"] = True
        return {
            "polo":         "empregado" if "Empregado" in polo else "empregador",
            "tipo_vinculo": mapa_vinculo[tipo_vinculo],
        }

    st.info("👈 Preencha seu perfil na barra lateral para começar.")
    return None

@st.cache_resource
def carregar_vectorstore():
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    if not os.path.exists(PERSIST_DIRECTORY):
        st.error("⚠️ Banco vetorial não encontrado. Rode primeiro: python indexar.py")
        st.stop()

    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    return Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings
    )

def buscar_documentos(pergunta: str, vectorstore, contexto: dict, k: int = 8):
    filtro = {}
    tipo = contexto.get("tipo_vinculo", "geral")

    if tipo != "geral":
        filtro["tipo_trabalhador"] = {"$in": [tipo, "clt_geral", "geral"]}

    if filtro:
        return vectorstore.similarity_search(pergunta, k=k, filter=filtro)
    return vectorstore.similarity_search(pergunta, k=k)

def rerank_documentos(pergunta: str, documentos: list, client) -> list:
    trechos = ""
    for i, doc in enumerate(documentos, 1):
        fonte  = doc.metadata.get("id_legislacao", "N/A")
        artigo = doc.metadata.get("artigo", "N/A")
        trechos += f"\n[{i}] {fonte} — {artigo}:\n{doc.page_content[:400]}\n"

    prompt = f"""Você é um avaliador jurídico trabalhista.
Dada a pergunta abaixo, avalie cada trecho numerado de 0 a 10 por relevância.
0 = irrelevante, 10 = responde diretamente a pergunta.

Pergunta: "{pergunta}"

Trechos:
{trechos}

Responda APENAS neste formato, um por linha, sem explicações:
1: <nota>
2: <nota>
3: <nota>"""

    response = client.models.generate_content(model=LLM_MODEL, contents=prompt)

    notas = {}
    for linha in response.text.strip().split("\n"):
        match = re.match(r'(\d+):\s*([\d.]+)', linha)
        if match:
            notas[int(match.group(1))] = float(match.group(2))

    indexados = [(i + 1, doc) for i, doc in enumerate(documentos)]
    return [doc for _, doc in sorted(indexados, key=lambda x: notas.get(x[0], 0), reverse=True)]

def gerar_resposta(pergunta: str, documentos: list, contexto: dict, client: genai.Client) -> str:
    polo_label = "empregado/trabalhador" if contexto["polo"] == "empregado" else "empregador/empresa"

    contexto_juridico = ""
    for i, doc in enumerate(documentos[:5], 1):
        fonte  = doc.metadata.get("id_legislacao", "N/A")
        artigo = doc.metadata.get("artigo", "N/A")
        contexto_juridico += f"\n[{i}] {fonte} — {artigo}:\n{doc.page_content[:600]}\n"

    prompt = f"""Você é um assistente jurídico trabalhista especializado.
Responda à pergunta abaixo de forma clara e objetiva, citando os artigos e leis relevantes.

Perfil do usuário: {polo_label} — vínculo: {contexto['tipo_vinculo']}

Pergunta: "{pergunta}"

Legislação relevante encontrada:
{contexto_juridico}

Instruções:
- SEMPRE informe brevemente que você é uma IA e que não é um profissional formado em direito.
- SEMPRE mantenha o idioma português, mesmo que a pergunta seja em outro idioma.
- Cite os artigos no formato: (Art. X — Nome da Lei)
- Indique claramente os direitos e obrigações para o perfil informado
- Se houver diferença de tratamento entre empregado e empregador, destaque
- Seja objetivo, evite jargão excessivo"""

    response = client.models.generate_content(model=LLM_MODEL, contents=prompt)
    return response.text

def main():
    st.set_page_config(page_title="Agente Jurídico Trabalhista", page_icon="⚖️")
    st.title("⚖️ Agente Jurídico Trabalhista")

    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

    if "historico" not in st.session_state:
        st.session_state["historico"] = []

    contexto = coletar_contexto_usuario()
    if not contexto:
        return

    vs = carregar_vectorstore()

    # Exibe histórico
    for msg in st.session_state["historico"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Input da pergunta
    pergunta = st.chat_input("Digite sua dúvida trabalhista...")
    if not pergunta:
        return

    with st.chat_message("user"):
        st.markdown(pergunta)
    st.session_state["historico"].append({"role": "user", "content": pergunta})

    with st.chat_message("assistant"):
        with st.status("Consultando legislação...", expanded=True) as status:

            st.write("📚 Step 2 — Buscando legislação relevante...")
            docs_recuperados = buscar_documentos(pergunta, vs, contexto)
            st.caption(f"{len(docs_recuperados)} trechos encontrados")

            st.write("⚖️ Step 3 — Reordenando por relevância...")
            docs_reranked = rerank_documentos(pergunta, docs_recuperados, client)

            st.write("✍️ Step 4 — Gerando resposta com citações...")
            resposta = gerar_resposta(pergunta, docs_reranked, contexto, client)

            status.update(label="✅ Consulta concluída!", state="complete")

        st.markdown(resposta)

        with st.expander("📎 Fontes consultadas"):
            for doc in docs_reranked[:5]:
                st.markdown(
                    f"**{doc.metadata.get('id_legislacao')}** — "
                    f"{doc.metadata.get('artigo')} "
                )

    st.session_state["historico"].append({"role": "assistant", "content": resposta})

if __name__ == "__main__":
    main()