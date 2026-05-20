import os
import re
import uuid
import streamlit as st
from langchain_community.vectorstores import Chroma
from google import genai
from google.genai.errors import ServerError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from langfuse import get_client

PERSIST_DIRECTORY  = "./chroma_rh"
EMBEDDING_MODEL    = "models/gemini-embedding-001"
LLM_MODEL          = "gemini-2.5-flash"
LLM_MODEL_FALLBACK = "gemini-2.5-flash-lite"
BASE_DIR           = os.path.dirname(os.path.abspath(__file__))

os.environ["LANGFUSE_PUBLIC_KEY"] = st.secrets["LANGFUSE_PUBLIC_KEY"]
os.environ["LANGFUSE_SECRET_KEY"] = st.secrets["LANGFUSE_SECRET_KEY"]
os.environ["LANGFUSE_HOST"]       = st.secrets.get("LANGFUSE_HOST", "https://cloud.langfuse.com")
os.environ["GOOGLE_API_KEY"]      = st.secrets["GOOGLE_API_KEY"]

@st.cache_resource
def get_langfuse():
    """Cliente singleton thread-safe, conforme docs do Langfuse v3."""
    return get_client()

@st.cache_resource
def get_genai_client():
    return genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

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

@retry(
    retry=retry_if_exception_type(ServerError),
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=2, min=3, max=20),
    reraise=True,
)
def _gerar_com_modelo(client, prompt: str, modelo: str) -> str:
    """Chama o Gemini e registra como `generation` no trace atual."""
    langfuse = get_langfuse()
    with langfuse.start_as_current_observation(
        as_type="generation",
        name="gemini_call",
        model=modelo,
        input=prompt,
    ) as generation:
        response = client.models.generate_content(model=modelo, contents=prompt)
        texto = response.text

        usage = getattr(response, "usage_metadata", None)
        if usage:
            generation.update(
                output=texto,
                usage_details={
                    "input":  getattr(usage, "prompt_token_count", 0),
                    "output": getattr(usage, "candidates_token_count", 0),
                    "total":  getattr(usage, "total_token_count", 0),
                },
            )
        else:
            generation.update(output=texto)

        return texto

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

def buscar_documentos(pergunta: str, vectorstore, contexto: dict, k: int = 8):
    langfuse = get_langfuse()
    tipo = contexto.get("tipo_vinculo", "geral")

    with langfuse.start_as_current_observation(
        as_type="span",
        name="retrieval",
        input={"pergunta": pergunta, "k": k, "tipo_vinculo": tipo},
    ) as span:
        if tipo == "pj":
            filtro = {"tipo_trabalhador": {"$in": ["pj", "geral"]}}
            docs = vectorstore.similarity_search(pergunta, k=k, filter=filtro)
        elif tipo != "geral":
            filtro = {"tipo_trabalhador": {"$in": [tipo, "clt_geral", "geral"]}}
            docs = vectorstore.similarity_search(pergunta, k=k, filter=filtro)
        else:
            docs = vectorstore.similarity_search(pergunta, k=k)

        span.update(output={
            "n_docs": len(docs),
            "fontes": [
                {
                    "id_legislacao": d.metadata.get("id_legislacao"),
                    "artigo":        d.metadata.get("artigo"),
                }
                for d in docs
            ],
        })
        return docs

def rerank_documentos(pergunta: str, documentos: list, client) -> list:
    langfuse = get_langfuse()

    with langfuse.start_as_current_observation(
        as_type="span",
        name="rerank",
        input={"pergunta": pergunta, "n_docs_entrada": len(documentos)},
    ) as span:
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
        reordenado = [doc for _, doc in sorted(indexados, key=lambda x: notas.get(x[0], 0), reverse=True)]

        span.update(output={
            "notas": notas,
            "ordem_final": [d.metadata.get("artigo") for d in reordenado[:5]],
        })
        return reordenado

def gerar_resposta(pergunta: str, documentos: list, contexto: dict, client) -> str:
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

def processar_pergunta(pergunta: str, vs, contexto: dict, client, session_id: str):
    """Orquestra retrieval -> rerank -> generation dentro de um único trace."""
    langfuse = get_langfuse()

    with langfuse.start_as_current_observation(
        as_type="span",
        name="pipeline_juridico",
        input=pergunta,
    ) as trace_root:
        langfuse.update_current_trace(
            session_id=session_id,
            user_id=f"{contexto['polo']}__{contexto['tipo_vinculo']}",
            tags=[contexto["polo"], contexto["tipo_vinculo"]],
            metadata={
                "polo": contexto["polo"],
                "tipo_vinculo": contexto["tipo_vinculo"],
            },
        )

        docs            = buscar_documentos(pergunta, vs, contexto)
        docs_reranked   = rerank_documentos(pergunta, docs, client)
        resposta        = gerar_resposta(pergunta, docs_reranked, contexto, client)

        trace_root.update(output=resposta)
        return resposta, docs_reranked

def main():
    st.set_page_config(page_title="Agente Jurídico Trabalhista", page_icon="⚖️")

    LIMITE_PERGUNTAS = 5

    if "total_perguntas" not in st.session_state:
        st.session_state["total_perguntas"] = 0

    if "langfuse_session_id" not in st.session_state:
        st.session_state["langfuse_session_id"] = str(uuid.uuid4())

    if st.session_state["total_perguntas"] >= LIMITE_PERGUNTAS:
        st.warning("⚠️ Limite de perguntas desta sessão atingido. Reabra o app para continuar.")
        return

    st.title("⚖️ Agente Jurídico Trabalhista")
    st.caption(
        "⚠️ Ferramenta de informação jurídica — não substitui consultoria de advogado habilitado."
    )

    client    = get_genai_client()
    langfuse  = get_langfuse()

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
                resposta, docs_reranked = processar_pergunta(
                    pergunta,
                    vs,
                    contexto,
                    client,
                    session_id=st.session_state["langfuse_session_id"],
                )
            except ServerError:
                st.error(
                    "⚠️ O serviço do Gemini está sobrecarregado no momento. "
                    "Tente novamente em alguns minutos."
                )
                return
            finally:
                langfuse.flush()

        st.markdown(resposta)

        with st.expander("📎 Fontes consultadas"):
            for doc in docs_reranked[:5]:
                st.markdown(
                    f"**{doc.metadata.get('id_legislacao')}** — "
                    f"{doc.metadata.get('artigo')} "
                )

    st.session_state["historico"].append({"role": "assistant", "content": resposta})
    st.session_state["total_perguntas"] += 1

if __name__ == "__main__":
    main()