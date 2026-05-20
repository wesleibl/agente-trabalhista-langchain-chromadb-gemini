import os
import re
import json
import shutil
import requests
from bs4 import BeautifulSoup
from datetime import datetime
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

BASE_DIR          = os.path.dirname(os.path.abspath(__file__))
PERSIST_DIRECTORY = "./chroma_rh"
EMBEDDING_MODEL   = "models/gemini-embedding-001"
REGISTRO_PATH     = "./chroma_rh/registro.json"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

FONTES_URL = {
    "constituicao.htm":        "https://www.planalto.gov.br/ccivil_03/constituicao/constituicao.htm",
    "decreto-lei-del5452.htm": "https://www.planalto.gov.br/ccivil_03/decreto-lei/del5452compilado.htm",
    "estagio.htm":             "https://www.planalto.gov.br/ccivil_03/_ato2007-2010/2008/lei/l11788.htm",
    "fgts-8036.htm":           "https://www.planalto.gov.br/ccivil_03/leis/l8036consol.htm",
    "previdencia-social.htm":  "https://www.planalto.gov.br/ccivil_03/leis/l8213cons.htm",
    "tercerizada.htm":         "https://www.planalto.gov.br/ccivil_03/_ato2015-2018/2017/lei/l13429.htm",
    "trabalho-domestico.htm":  "https://www.planalto.gov.br/ccivil_03/leis/lcp/lcp150.htm",
    "codigo-civil.htm": "https://www.planalto.gov.br/ccivil_03/leis/2002/l10406compilada.htm",

}

FONTE_POR_ARQUIVO = {
    "constituicao.htm":        ("Constituicao_Federal", "CF/1988",                "clt_geral"),
    "decreto-lei-del5452.htm": ("CLT",                  "Decreto-Lei 5.452/1943", "clt_geral"),
    "estagio.htm":             ("Lei_do_Estagio",        "Lei 11.788/2008",        "estagiario"),
    "fgts-8036.htm":           ("Lei_do_FGTS",           "Lei 8.036/1990",         "clt_geral"),
    "previdencia-social.htm":  ("Lei_Previdencia",       "Lei 8.213/1991",         "geral"),
    "tercerizada.htm":         ("Lei_Terceirizacao",     "Lei 13.429/2017",        "terceirizado"),
    "trabalho-domestico.htm":  ("Lei_Domestico",         "LC 150/2015",            "domestico"),
    "codigo-civil.htm":        ("Codigo_Civil",          "Lei 10.406/2002",        "pj"),

}

MAPA_TEMATICO = {
    "beneficios": {
        "ferias":          ["ferias", "gozo", "abono pecuniario", "decimo terceiro"],
        "vale_transporte": ["vale-transporte", "transporte", "deslocamento"],
        "alimentacao":     ["vale-refeicao", "vale-alimentacao", "refeicao"],
    },
    "jornada_de_trabalho": {
        "horas_extras":   ["hora extra", "sobrejornada", "servico extraordinario", "jornada suplementar"],
        "banco_de_horas": ["banco de horas", "compensacao"],
        "intervalos":     ["intervalo", "intrajornada", "interjornada", "descanso"],
    },
    "vinculo_empregaticio": {
        "subordinacao":   ["subordinacao", "subordinado", "dependencia"],
        "habitualidade":  ["habitualidade", "nao eventual", "continuidade"],
        "pessoalidade":   ["pessoalidade", "pessoal"],
        "pejotizacao":    ["fraude", "vinculo empregaticio", "reconhecimento de vinculo"],
    },
    "modalidades_contrato": {
        "home_office": ["home office", "teletrabalho", "hibrido"],
        "estagio":     ["estagio", "estagiario"],
        "pj":          ["prestacao de servicos", "pessoa juridica", "autonomo"],
    },
    "extincao_contratual": {
        "justa_causa":  ["justa causa", "falta grave", "indisciplina", "art. 482"],
        "aviso_previo": ["aviso previo", "indenizado"],
        "rescisao":     ["rescisao", "verbas rescisorias", "demissao"],
    },
    "encargos_e_tributos": {
        "fgts": ["fgts", "fundo de garantia", "multa de 40"],
    },
    
}

def ler_registro() -> dict:
    """
    Retorna dict com nome do arquivo e data de indexação.
    Ex: {"estagio.htm": "2026-05-14 10:32:00"}
    """
    if not os.path.exists(REGISTRO_PATH):
        return {}
    with open(REGISTRO_PATH, "r") as f:
        return json.load(f)

def registrar_arquivos(arquivos: list):
    """Salva nome e timestamp dos arquivos indexados."""
    registro = ler_registro()
    agora    = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for arquivo in arquivos:
        registro[arquivo] = agora
    os.makedirs(PERSIST_DIRECTORY, exist_ok=True)
    with open(REGISTRO_PATH, "w") as f:
        json.dump(registro, f, indent=2, ensure_ascii=False)

def carregar_documentos(chaves: list) -> list:
    documentos = []
    total = len(chaves)

    for i, chave in enumerate(chaves, 1):
        url = FONTES_URL[chave]
        print(f"[{i}/{total}] Baixando: {chave}...")

        try:
            resp = requests.get(url, headers=HEADERS, timeout=30)
            resp.raise_for_status()

            soup       = BeautifulSoup(resp.text, "lxml")
            texto      = soup.get_text(separator="\n")
            linhas     = [l.strip() for l in texto.splitlines() if l.strip()]
            texto_limpo = "\n".join(linhas)

            print(f"       ✓ {len(texto_limpo)} caracteres\n")

            documentos.append(Document(
                page_content=texto_limpo,
                metadata={"documento": chave}
            ))

        except Exception as e:
            print(f"       ✗ ERRO: {e} — pulando\n")
            continue

    print(f"Total carregado: {len(documentos)} documentos\n")
    return documentos

def chunk_por_artigo(documentos: list) -> list:
    print("Dividindo por artigos...")
    chunks = []
    for doc in documentos:
        matches = list(re.finditer(
            r'(Art\.?\s*\d+[º°]?[\s\S]*?)(?=Art\.?\s*\d+|$)',
            doc.page_content, re.IGNORECASE
        ))
        for match in matches:
            conteudo = match.group(1).strip()
            if len(conteudo) < 30:
                continue
            num = re.search(r'Art\.?\s*(\d+)', conteudo, re.IGNORECASE)
            chunks.append(Document(
                page_content=conteudo,
                metadata={
                    **doc.metadata,
                    "artigo": f"Art. {num.group(1)}" if num else "N/A"
                }
            ))
    print(f"Total: {len(chunks)} chunks\n")
    return chunks


def enriquecer_chunks(chunks: list) -> list:
    print("Enriquecendo metadados...")
    for chunk in chunks:
        texto = chunk.page_content.lower()

        contagem_macro        = {}
        primeira_subcategoria = "geral"

        for macro, micros in MAPA_TEMATICO.items():
            matches = 0
            for micro, palavras in micros.items():
                if any(p in texto for p in palavras):
                    matches += 1
                    if primeira_subcategoria == "geral":
                        primeira_subcategoria = micro
            contagem_macro[macro] = matches

        categoria_principal = (
            max(contagem_macro, key=contagem_macro.get)
            if max(contagem_macro.values()) > 0
            else "geral"
        )

        chunk.metadata["categoria_principal"] = categoria_principal
        chunk.metadata["subcategoria"]        = primeira_subcategoria

        nome_arquivo             = chunk.metadata.get("documento", "").lower()
        fonte, id_leg, tipo_trab = ("Outros", "N/A", "geral")
        for chave, valores in FONTE_POR_ARQUIVO.items():
            if chave in nome_arquivo:
                fonte, id_leg, tipo_trab = valores
                break

        chunk.metadata["fonte_documento"]  = fonte
        chunk.metadata["id_legislacao"]    = id_leg
        chunk.metadata["tipo_trabalhador"] = tipo_trab

        match = re.search(r'(?:art\.?|artigo)\s*(\d+)', chunk.page_content, re.IGNORECASE)
        if match:
            chunk.metadata["artigo"] = f"Art. {match.group(1)}"

        chunk.metadata["polo_foco"]       = "empregado"
        chunk.metadata["ano_atualizacao"] = 2026

    print("Metadados enriquecidos.\n")
    return chunks

def indexar():
    todos_caminhos = list(FONTES_URL.keys())
    registro       = ler_registro()
    ja_indexados   = set(registro.keys())

    # ── Banco não existe: cria do zero ──
    if not os.path.exists(PERSIST_DIRECTORY) or not ja_indexados:
        print("Nenhum banco encontrado. Criando do zero...\n")
        chaves_para_processar = todos_caminhos

    # ── Banco existe: oferece opções ──
    else:
        pendentes = [c for c in todos_caminhos if c not in ja_indexados]

        print(f"✅ Banco existente em '{PERSIST_DIRECTORY}'")
        print(f"\n   Arquivos já indexados ({len(ja_indexados)}):")
        for a in sorted(ja_indexados):
            print(f"     • {a:40s} indexado em: {registro[a]}")

        if pendentes:
            print(f"\n   Arquivos ainda NÃO indexados ({len(pendentes)}):")
            for a in pendentes:
                print(f"     • {a}")

        print("\n   Opções:")
        print("   [1] Adicionar apenas os arquivos pendentes")
        print("   [2] Escolher arquivos específicos para adicionar")
        print("   [3] Reindexar tudo do zero")
        print("   [4] Cancelar")
        escolha = input("\n   Escolha (1/2/3/4): ").strip()

        if escolha == "4" or escolha == "":
            print("Cancelado.")
            return

        elif escolha == "1":
            if not pendentes:
                print("\nTodos os arquivos já estão indexados. Nada a fazer.")
                return
            chaves_para_processar = pendentes

        elif escolha == "2":
            print("\n   Arquivos disponíveis:")
            for a in todos_caminhos:
                status = "✓ já indexado" if a in ja_indexados else "pendente"
                print(f"     • {a:40s} ({status})")
            entrada = input("\n   Digite os nomes separados por espaço: ").strip()
            chaves_para_processar = [c for c in entrada.split() if c in todos_caminhos]
            if not chaves_para_processar:
                print("Nenhum arquivo válido informado. Cancelado.")
                return

        elif escolha == "3":
            confirmacao = input("   ⚠️  Isso apaga o banco inteiro. Confirma? (s/N): ").strip().lower()
            if confirmacao != "s":
                print("Cancelado.")
                return
            shutil.rmtree(PERSIST_DIRECTORY)
            print("   Banco anterior removido.\n")
            chaves_para_processar = todos_caminhos

        else:
            print("Opção inválida. Cancelado.")
            return

    # ── Processa os arquivos escolhidos ──
    docs = carregar_documentos(chaves_para_processar)
    if not docs:
        print("Nenhum documento carregado. Verifique sua conexão.")
        return

    chunks = chunk_por_artigo(docs)
    if not chunks:
        print("Nenhum chunk gerado.")
        return

    chunks = enriquecer_chunks(chunks)

    print(f"Gerando embeddings para {len(chunks)} chunks...")
    print("Isso pode levar alguns minutos...\n")

    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)

    if os.path.exists(PERSIST_DIRECTORY) and ja_indexados and escolha != "3":
        # Adiciona ao banco existente
        vs = Chroma(
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=embeddings
        )
        vs.add_documents(chunks)
        print(f"\n✅ {len(chunks)} chunks adicionados ao banco existente.")
    else:
        # Cria banco novo
        Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=PERSIST_DIRECTORY
        )
        print(f"\n✅ Banco criado com sucesso em '{PERSIST_DIRECTORY}'")
        print(f"   {len(chunks)} chunks indexados.")

    # Registra os arquivos processados com timestamp
    registrar_arquivos(chaves_para_processar)

    print(f"\n📋 Registro atualizado:")
    registro_atualizado = ler_registro()
    for a in sorted(registro_atualizado):
        print(f"   • {a:40s} indexado em: {registro_atualizado[a]}")

    print(f"\nAgora rode: streamlit run app.py")

if __name__ == "__main__":
    indexar()