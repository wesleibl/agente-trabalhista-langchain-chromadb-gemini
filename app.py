import os
import re
import streamlit as st
from langchain_community.vectorstores import Chroma
from google import genai
from google.genai.errors import ServerError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from dotenv import load_dotenv

load_dotenv()

PERSIST_DIRECTORY  = "./chroma_rh"
EMBEDDING_MODEL    = "models/gemini-embedding-001"
LLM_MODEL          = "gemini-2.5-flash"
LLM_MODEL_FALLBACK = "gemini-2.5-flash-lite"
BASE_DIR           = os.path.dirname(os.path.abspath(__file__))

@retry(
    retry=retry_if_exception_type(ServerError),
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=2, min=3, max=20),
    reraise=True,
)
def _gerar_com_modelo(client, prompt: str, modelo: str) -> str:
    response = client.models.generate_content(model=modelo, contents=prompt)
    return response.text

def chamar_llm(client, prompt: str) -> str:
    try:
        return _gerar_com_modelo(client, prompt, LLM_MODEL)
    except ServerError:
        return _gerar_com_modelo(client, prompt, LLM_MODEL_FALLBACK)

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

        st.divider()
        st.warning(
            "**Aviso legal**\n\n"
            "Este assistente fornece **informações jurídicas gerais** com base na "
            "legislação vigente. Ele **não é um advogado** e **não substitui "
            "consultoria jurídica profissional**.\n\n"
            "Para orientação sobre seu caso concreto, consulte um advogado "
            "trabalhista habilitado na OAB.",
            icon="⚠️",
        )

    mapa_vinculo = {
        "CLT Geral":                 "clt_geral",
        "Doméstico":                 "domestico",
        "Estagiário":                "estagiario",
        "Terceirizado / Temporário": "terceirizado",
        "Autônomo / PJ":             "pj",
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
    tipo = contexto.get("tipo_vinculo", "geral")

    if tipo == "pj":
        filtro = {"tipo_trabalhador": {"$in": ["pj", "geral"]}}
        return vectorstore.similarity_search(pergunta, k=k, filter=filtro)

    if tipo != "geral":
        filtro = {"tipo_trabalhador": {"$in": [tipo, "clt_geral", "geral"]}}
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

    texto_resposta = chamar_llm(client, prompt)

    notas = {}
    for linha in texto_resposta.strip().split("\n"):
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

    prompt = f"""Você é um assistente jurídico trabalhista. Responda de forma clara, objetiva e fundamentada.

        Perfil do usuário: {polo_label} — vínculo: {contexto['tipo_vinculo']}

        Pergunta: "{pergunta}"

        Legislação encontrada:
        {contexto_juridico}

        ESTRUTURA DA RESPOSTA (siga esta ordem, em prosa fluida):

        1. Resposta direta (1-2 frases): responda objetivamente o que foi perguntado.
        2. Fundamentação legal (2-4 parágrafos curtos): explique a base jurídica citando artigos da legislação encontrada acima. Use o formato (Art. X — Nome da Lei).
        3. Orientação prática (1-2 frases): o que o usuário deve fazer ou observar na prática.

        REGRAS:
        - FOQUE no perfil do usuário ({contexto['tipo_vinculo']}). Mencione outros vínculos APENAS se for essencial para comparação direta.
        - Se o vínculo for "pj" e a pergunta envolver subordinação, horários, habitualidade ou rotina fixa, ALERTE sobre possível configuração de vínculo empregatício (Art. 3º da CLT) e os riscos da pejotização para ambas as partes.
        - Mantenha sempre o português, mesmo se a pergunta vier em outro idioma.
        - Use os artigos da legislação encontrada — não invente artigos.
        - NÃO mencione ao usuário se um artigo está ou não presente nos materiais fornecidos. Apenas cite os artigos relevantes naturalmente, como se fizessem parte do seu conhecimento jurídico.
        - NÃO use frases como "não presente nos materiais", "conforme o contexto fornecido", "com base nos trechos" ou similares. Responda como um especialista, sem expor a mecânica interna do sistema.
        - NÃO numere os blocos da resposta (não escreva "1. Resposta direta", "2. Fundamentação"). Use prosa fluida com parágrafos.
        - Seja objetivo. Evite jargão excessivo e textos longos demais.
        - Termine com UMA linha curta: "_Informação geral — para seu caso, consulte um advogado trabalhista._"
        - NÃO se apresente como IA — o app já informa isso ao usuário."""

    return chamar_llm(client, prompt)

def main():
    st.set_page_config(page_title="Agente Jurídico Trabalhista", page_icon="⚖️")
    st.title("⚖️ Agente Jurídico Trabalhista")
    st.caption(
        "⚠️ Ferramenta de informação jurídica — não substitui consultoria de advogado habilitado."
    )

    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

    if "historico" not in st.session_state:
        st.session_state["historico"] = []

    contexto = coletar_contexto_usuario()
    if not contexto:
        return

    vs = carregar_vectorstore()

    for msg in st.session_state["historico"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    pergunta = st.chat_input("Digite sua dúvida trabalhista...")
    if not pergunta:
        return

    with st.chat_message("user"):
        st.markdown(pergunta)
    st.session_state["historico"].append({"role": "user", "content": pergunta})

    with st.chat_message("assistant"):
        with st.spinner("Consultando legislação..."):
            try:
                docs_recuperados = buscar_documentos(pergunta, vs, contexto)
                docs_reranked    = rerank_documentos(pergunta, docs_recuperados, client)
                resposta         = gerar_resposta(pergunta, docs_reranked, contexto, client)
            except ServerError:
                st.error(
                    "⚠️ O serviço do Gemini está sobrecarregado no momento. "
                    "Tente novamente em alguns minutos."
                )
                return

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