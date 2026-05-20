import os
import re
import uuid
import streamlit as st
from langchain_community.vectorstores import Chroma
from google import genai
from google.genai.errors import ServerError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from langfuse import get_client, propagate_attributes

PERSIST_DIRECTORY  = "./chroma_rh"
EMBEDDING_MODEL    = "models/gemini-embedding-001"
LLM_MODEL          = "gemini-2.5-flash"
LLM_MODEL_FALLBACK = "gemini-2.5-flash-lite"
BASE_DIR           = os.path.dirname(os.path.abspath(__file__))

os.environ["LANGFUSE_PUBLIC_KEY"] = st.secrets["LANGFUSE_PUBLIC_KEY"]
os.environ["LANGFUSE_SECRET_KEY"] = st.secrets["LANGFUSE_SECRET_KEY"]
os.environ["LANGFUSE_HOST"]       = st.secrets.get("LANGFUSE_HOST", "https://cloud.langfuse.com")
os.environ["GOOGLE_API_KEY"]      = st.secrets["GOOGLE_API_KEY"]

MAPA_VINCULO = {
    "CLT Geral":                 "clt_geral",
    "Doméstico":                 "domestico",
    "Estagiário":                "estagiario",
    "Terceirizado / Temporário": "terceirizado",
    "Autônomo / PJ":             "pj",
}

POLOS = {
    "Empregado / Trabalhador":   "empregado",
    "Empregador / Empresa / RH": "empregador",
}

RESPOSTAS_ONBOARDING = set(list(POLOS.keys()) + list(MAPA_VINCULO.keys()))

PADROES_INJECTION = [
    r"ignore\s+(tudo|todas|todos|o\s+que|as\s+instru[cç][oõ]es)",
    r"esqueça\s+(tudo|o\s+que\s+foi|as\s+instru[cç][oõ]es|seu\s+papel)",
    r"(ignore|forget|disregard)\s+(previous|prior|all|above|everything)",
    r"(new|novo)\s+(prompt|instru[cç][aã]o|comando|role|papel|sistema)",
    r"act\s+as\s+(?!a\s+lawyer|um\s+advogado)",
    r"you\s+are\s+now",
    r"(pretend|finja|simule?)\s+(que\s+)?(you\s+are|voc[eê]\s+[eé]|ser)",
    r"system\s*:",
    r"<\s*system\s*>",
    r"\[system\]",
    r"###\s*(instru[cç][oõ]es|instructions|prompt|system)",
    r"(revele?|mostre?|exiba?|print|reveal|show)\s+(o\s+)?(prompt|instru[cç][oõ]es|system)",
    r"(mude?|altere?|troque?|change|override)\s+(seu\s+)?(comportamento|papel|role|instru[cç][oõ]es)",
    r"jailbreak",
    r"dan\s+mode",
    r"developer\s+mode",
    r"modo\s+(desenvolvedor|irrestrito|livre|admin)",
]

_RE_INJECTION = re.compile(
    "|".join(PADROES_INJECTION),
    flags=re.IGNORECASE | re.UNICODE,
)

def detectar_injection(texto: str) -> bool:
    return bool(_RE_INJECTION.search(texto))

def validar_pergunta(texto: str, historico: list) -> tuple[bool, str]:
    perguntas_anteriores = [
        m for m in historico
        if m["role"] == "user" and m["content"] not in RESPOSTAS_ONBOARDING
    ]

    if perguntas_anteriores:
        return True, ""

    texto_limpo = texto.strip()

    if len(texto_limpo) < 5:
        return False, "Pode detalhar um pouco mais sua dúvida? Preciso de mais contexto para consultar a legislação corretamente. 😊"

    if re.fullmatch(r'[\W_]+', texto_limpo, flags=re.UNICODE):
        return False, "Pode detalhar um pouco mais sua dúvida? Preciso de mais contexto para consultar a legislação corretamente. 😊"

    if len(re.findall(r'\b\w+\b', texto_limpo)) < 2:
        return False, "Pode detalhar um pouco mais sua dúvida? Preciso de mais contexto para consultar a legislação corretamente. 😊"

    return True, ""

def classificar_escopo(pergunta: str, client) -> str:
    prompt = f"""Você é um classificador. Responda APENAS com uma das três opções abaixo, sem pontuação, sem explicação.

        Categorias:
        - "trabalhista": a pergunta é sobre direito trabalhista, CLT, FGTS, férias, rescisão, salário, jornada, vínculo empregatício ou qualquer tema de legislação trabalhista brasileira.
        - "relacionado": a pergunta envolve uma situação pessoal ou social (pensão alimentícia, divórcio, herança, saúde, etc.) mas o contexto indica que a pessoa pode estar buscando apoio no ambiente de trabalho ou perguntando em nome de um funcionário/colega.
        - "fora_escopo": a pergunta não tem nenhuma relação com trabalho, funcionários ou legislação brasileira.

        Pergunta: "{pergunta}"

        Responda somente uma palavra: trabalhista, relacionado ou fora_escopo"""

    try:
        resposta = chamar_llm(client, prompt).strip().lower()
        if "trabalhista" in resposta:
            return "trabalhista"
        if "relacionado" in resposta:
            return "relacionado"
        return "fora_escopo"
    except Exception:
        return "trabalhista"

@st.cache_resource
def get_langfuse():
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
    langfuse = get_langfuse()

    with propagate_attributes(
        session_id=session_id,
        user_id=f"{contexto['polo']}__{contexto['tipo_vinculo']}",
        tags=[contexto["polo"], contexto["tipo_vinculo"]],
        metadata={
            "polo": contexto["polo"],
            "tipo_vinculo": contexto["tipo_vinculo"],
        },
    ):
        with langfuse.start_as_current_observation(
            as_type="span",
            name="pipeline_juridico",
            input=pergunta,
        ) as trace_root:
            docs          = buscar_documentos(pergunta, vs, contexto)
            docs_reranked = rerank_documentos(pergunta, docs, client)
            resposta      = gerar_resposta(pergunta, docs_reranked, contexto, client)

            trace_root.update(output=resposta)
            return resposta, docs_reranked


def inicializar_onboarding():
    if not st.session_state["historico"]:
        st.session_state["historico"].append({
            "role": "assistant",
            "content": (
                "Olá! 👋 Sou um assistente jurídico trabalhista. "
                "Antes de começar, preciso entender seu perfil para dar respostas mais precisas.\n\n"
                "**Você é empregado ou empregador?**"
            ),
        })

def renderizar_botoes_polo():
    col1, col2 = st.columns(2)
    with col1:
        if st.button("👷 Empregado / Trabalhador", use_container_width=True, key="btn_empregado"):
            return "Empregado / Trabalhador"
    with col2:
        if st.button("🏢 Empregador / Empresa / RH", use_container_width=True, key="btn_empregador"):
            return "Empregador / Empresa / RH"
    return None

def renderizar_botoes_vinculo():
    opcoes = list(MAPA_VINCULO.keys())
    cols = st.columns(2)
    escolhido = None
    for i, opcao in enumerate(opcoes):
        with cols[i % 2]:
            if st.button(opcao, use_container_width=True, key=f"btn_vinculo_{i}"):
                escolhido = opcao
    return escolhido

def avancar_para_vinculo(polo_label: str):
    st.session_state["polo_confirmado"] = POLOS[polo_label]
    st.session_state["historico"].append({"role": "user", "content": polo_label})
    st.session_state["historico"].append({
        "role": "assistant",
        "content": (
            "Perfeito! Agora me diga: **qual é o tipo de vínculo?**\n\n"
            "Isso me ajuda a focar nas leis certas para o seu caso."
        ),
    })
    st.session_state["estado"] = "onboarding_vinculo"

def avancar_para_conversa(vinculo_label: str):
    st.session_state["vinculo_confirmado"]       = MAPA_VINCULO[vinculo_label]
    st.session_state["vinculo_label_confirmado"] = vinculo_label
    st.session_state["historico"].append({"role": "user", "content": vinculo_label})

    polo_humano = "trabalhador" if st.session_state["polo_confirmado"] == "empregado" else "empregador"
    st.session_state["historico"].append({
        "role": "assistant",
        "content": (
            f"Anotado: **{polo_humano}** com vínculo **{vinculo_label}**.\n\n"
            "Agora pode mandar sua dúvida trabalhista. Vou consultar a legislação "
            "e te dar uma resposta fundamentada. 📚"
        ),
    })
    st.session_state["estado"] = "conversando"

def trocar_perfil():
    st.session_state["historico"] = []
    st.session_state["estado"]    = "onboarding_polo"
    st.session_state.pop("polo_confirmado", None)
    st.session_state.pop("vinculo_confirmado", None)
    st.session_state.pop("vinculo_label_confirmado", None)
    inicializar_onboarding()


def main():
    st.set_page_config(
        page_title="Agente Jurídico Trabalhista",
        page_icon="⚖️",
        initial_sidebar_state="collapsed",
    )

    LIMITE_PERGUNTAS = 5

    if "total_perguntas" not in st.session_state:
        st.session_state["total_perguntas"] = 0
    if "estado" not in st.session_state:
        st.session_state["estado"] = "onboarding_polo"
    if "historico" not in st.session_state:
        st.session_state["historico"] = []
    if "langfuse_session_id" not in st.session_state:
        st.session_state["langfuse_session_id"] = str(uuid.uuid4())

    st.title("⚖️ Agente Jurídico Trabalhista")
    st.caption(
        "⚠️ Ferramenta de informação jurídica. Não substitui consultoria de advogado habilitado."
    )

    inicializar_onboarding()

    if st.session_state["estado"] == "conversando":
        polo_label    = "Empregado" if st.session_state["polo_confirmado"] == "empregado" else "Empregador"
        vinculo_label = st.session_state["vinculo_label_confirmado"]
        col1, col2 = st.columns([4, 1])
        with col1:
            st.caption(f"👤 **{polo_label}** · 📄 **{vinculo_label}**")
        with col2:
            if st.button("🔄 Trocar", use_container_width=True, key="btn_trocar_perfil"):
                trocar_perfil()
                st.rerun()
        st.divider()

    for msg in st.session_state["historico"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if st.session_state["estado"] == "onboarding_polo":
        escolha = renderizar_botoes_polo()
        if escolha:
            avancar_para_vinculo(escolha)
            st.rerun()
        with st.expander("⚠️ Aviso legal importante"):
            st.markdown(
                "Este assistente fornece **informações jurídicas gerais** com base na "
                "legislação vigente. Ele **não é um advogado** e **não substitui "
                "consultoria jurídica profissional**.\n\n"
                "Para orientação sobre seu caso concreto, consulte um advogado "
                "trabalhista habilitado na OAB."
            )
        return

    if st.session_state["estado"] == "onboarding_vinculo":
        escolha = renderizar_botoes_vinculo()
        if escolha:
            avancar_para_conversa(escolha)
            st.rerun()
        return

    if st.session_state["total_perguntas"] >= LIMITE_PERGUNTAS:
        st.warning("⚠️ Limite de perguntas desta sessão atingido. Reabra o app para continuar.")
        return

    contexto = {
        "polo":         st.session_state["polo_confirmado"],
        "tipo_vinculo": st.session_state["vinculo_confirmado"],
    }

    client   = get_genai_client()
    langfuse = get_langfuse()
    vs       = carregar_vectorstore()

    pergunta = st.chat_input("Digite sua dúvida trabalhista...")
    if not pergunta:
        return

    valida, motivo = validar_pergunta(pergunta, st.session_state["historico"])
    if not valida:
        with st.chat_message("user"):
            st.markdown(pergunta)
        st.session_state["historico"].append({"role": "user", "content": pergunta})
        with st.chat_message("assistant"):
            st.markdown(motivo)
        st.session_state["historico"].append({"role": "assistant", "content": motivo})
        return

    if detectar_injection(pergunta):
        with st.chat_message("user"):
            st.markdown(pergunta)
        st.session_state["historico"].append({"role": "user", "content": pergunta})
        msg = (
            "Só consigo responder dúvidas sobre legislação trabalhista. "
            "Se tiver uma pergunta sobre direitos ou obrigações no trabalho, pode mandar! 👍"
        )
        with st.chat_message("assistant"):
            st.markdown(msg)
        st.session_state["historico"].append({"role": "assistant", "content": msg})
        return

    with st.spinner("Verificando pergunta..."):
        escopo = classificar_escopo(pergunta, client)

    if escopo == "fora_escopo":
        msg = (
            "Só consigo ajudar com dúvidas sobre legislação trabalhista. "
            "Se tiver uma pergunta sobre direitos ou obrigações no trabalho, pode mandar! 👍"
        )
    elif escopo == "relacionado":
        msg = (
            "Essa situação envolve direito de família, que está fora da minha área. "
            "Para questões de pensão alimentícia, o caminho é buscar orientação com um advogado de família "
            "ou na Defensoria Pública, que oferece atendimento gratuito. 🙏\n\n"
            "Se surgir alguma dúvida sobre os direitos trabalhistas da sua funcionária em si, estou aqui!"
        )
    else:
        msg = None

    if msg:
        with st.chat_message("user"):
            st.markdown(pergunta)
        st.session_state["historico"].append({"role": "user", "content": pergunta})
        with st.chat_message("assistant"):
            st.markdown(msg)
        st.session_state["historico"].append({"role": "assistant", "content": msg})
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