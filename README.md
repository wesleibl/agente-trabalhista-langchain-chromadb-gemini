# ⚖️ Agente Jurídico Trabalhista

Assistente conversacional de direito do trabalho construído sobre uma pipeline **RAG** (Retrieval-Augmented Generation) completa. O sistema scrapa, estrutura e indexa legislação trabalhista brasileira diretamente do Planalto, e responde dúvidas jurídicas com citação dos artigos relevantes — adaptando o tom e o foco ao perfil do usuário (empregado ou empregador).

---

## Teste

Teste diretamente por este link: [agente-trabalhista](https://agente-trabalhista.streamlit.app/)

## Demonstração

```
Usuário: Tenho direito a horas extras trabalhando em home office?

Agente: Sim. O regime de teletrabalho não afasta o direito a horas extras
quando há controle de jornada, conforme:
• Art. 75-A ao 75-F — CLT (Decreto-Lei 5.452/1943)
• Art. 7°, XIII — CF/1988
```

---

## Arquitetura

```
┌─────────────────────────────────────────────────────────┐
│                    PIPELINE DE INDEXAÇÃO                │
│                      (indexar.py)                       │
│                                                         │
│  planalto.gov.br  ──►  BeautifulSoup  ──►  chunk        │
│                           scraping          por artigo  │
│                                               │         │
│                      enriquecimento           │         │
│                       de metadados  ◄─────────┘         │
│                            │                            │
│                      Gemini Embeddings                  │
│                            │                            │
│                        ChromaDB  ────────► disco        │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                  PIPELINE DE CONSULTA                   │
│                       (app.py)                          │
│                                                         │
│  Usuário  ──►  Perfil  ──►  Busca vetorial filtrada     │
│                              (ChromaDB + LangChain)     │
│                                     │                   │
│                              LLM Reranking              │
│                           (Gemini 2.5 Flash)            │
│                                     │                   │
│                           Geração de resposta           │
│                        com citações de artigos          │
│                                     │                   │
│                          Streamlit Chat UI              │
└─────────────────────────────────────────────────────────┘
```

---

## Tecnologias

| Camada | Tecnologia | Papel |
|---|---|---|
| **LLM** | Google Gemini 2.5 Flash (fallback: Flash Lite) | Reranking e geração de respostas |
| **Embeddings** | Gemini Embedding 001 | Vetorização dos chunks de legislação |
| **Vector Store** | ChromaDB (local persistente) | Armazenamento e busca semântica |
| **Orquestração** | LangChain Community + Core | Integração LLM/VectorStore/Documents |
| **Interface** | Streamlit | Chat UI reativa com sidebar de perfil |
| **Web Scraping** | requests + BeautifulSoup (lxml) | Download e limpeza da legislação |

---

## Estratégias e Decisões de Design

### 1. Chunking por Artigo Legal
Em vez de usar splitters genéricos por tamanho de token, a legislação é dividida **artigo por artigo** com regex (`Art\.?\s*\d+[º°]?`). Isso preserva a unidade semântica do direito: cada chunk carrega um artigo completo e pode ser citado de forma direta na resposta.

### 2. Enriquecimento de Metadados
Cada chunk recebe um conjunto rico de metadados no momento da indexação:

```python
{
  "fonte_documento":    "CLT",
  "id_legislacao":      "Decreto-Lei 5.452/1943",
  "tipo_trabalhador":   "clt_geral",   # clt_geral | domestico | estagiario | terceirizado | geral
  "categoria_principal": "jornada_de_trabalho",
  "subcategoria":       "horas_extras",
  "artigo":             "Art. 59",
  "polo_foco":          "empregado",
  "ano_atualizacao":    2026
}
```

### 3. Mapa Temático (Classificação Automática)
Um dicionário hierárquico de palavras-chave classifica cada chunk em macrocategorias e subcategorias jurídicas:

```
beneficios            → ferias | vale_transporte | alimentacao
jornada_de_trabalho   → horas_extras | banco_de_horas | intervalos
vinculo_empregaticio  → subordinacao | habitualidade | pessoalidade | pejotizacao
modalidades_contrato  → home_office | estagio | pj
extincao_contratual   → justa_causa | aviso_previo | rescisao
encargos_e_tributos   → fgts
```

### 4. Busca Vetorial com Filtro por Perfil
A busca semântica no ChromaDB usa os metadados para filtrar automaticamente documentos relevantes ao vínculo do usuário. Um estagiário nunca recebe como contexto artigos exclusivos de domésticos:

```python
filtro["tipo_trabalhador"] = {"$in": [tipo, "clt_geral", "geral"]}
```

### 5. LLM Reranking
Após a recuperação vetorial (top-8), um segundo prompt ao Gemini avalia cada trecho de 0 a 10 por relevância à pergunta. Os top-5 trechos reordenados alimentam a geração final — reduzindo ruído sem custo de embeddings adicionais.

### 6. Geração com Contexto de Perfil
O prompt de geração é personalizado com o perfil do usuário (polo e tipo de vínculo), instruindo o modelo a destacar direitos e obrigações específicos para aquele contexto e a citar artigos no formato padrão `(Art. X — Nome da Lei)`.

### 7. Indexação Incremental com Registro
`registro.json` rastreia quais fontes já foram indexadas e quando. Na próxima execução de `indexar.py`, o usuário pode adicionar apenas arquivos novos, reindexar fontes específicas ou recriar o banco do zero — sem reprocessar o que já existe.

### 8. Resiliência com Retry e Fallback de Modelo
Chamadas ao Gemini são protegidas por retry automático com backoff exponencial (`tenacity`): até 4 tentativas, aguardando entre 3 s e 20 s entre elas. Se o modelo primário (`gemini-2.5-flash`) continuar falhando, a aplicação troca automaticamente para o fallback (`gemini-2.5-flash-lite`). Erros persistentes são exibidos ao usuário sem travar a sessão.

### 9. Cache de Recursos com Streamlit
`@st.cache_resource` garante que o vector store (ChromaDB + embeddings) seja carregado uma única vez por sessão do servidor, evitando overhead em cada interação do chat.

### 10. Histórico de Conversa por Sessão
O histórico da conversa é mantido em `st.session_state["historico"]`, exibindo a troca completa a cada rerun do Streamlit e permitindo continuidade no diálogo.

---

## Fontes de Legislação

Todas as fontes são obtidas diretamente do [Portal da Legislação do Planalto](https://www.planalto.gov.br):

| Arquivo | Legislação | Vínculo |
|---|---|---|
| `constituicao.htm` | Constituição Federal — CF/1988 | Todos |
| `decreto-lei-del5452.htm` | CLT — Decreto-Lei 5.452/1943 | CLT Geral |
| `estagio.htm` | Lei do Estágio — Lei 11.788/2008 | Estagiário |
| `fgts-8036.htm` | Lei do FGTS — Lei 8.036/1990 | CLT Geral |
| `previdencia-social.htm` | Previdência Social — Lei 8.213/1991 | Todos |
| `tercerizada.htm` | Lei da Terceirização — Lei 13.429/2017 | Terceirizado |
| `trabalho-domestico.htm` | Trabalho Doméstico — LC 150/2015 | Doméstico |
| `codigo-civil.htm` | Código Civil — Lei 10.406/2002 | Autônomo / PJ |

---

## Instalação e Uso

### Pré-requisitos
- Python 3.10+
- Chave de API do Google AI Studio (Gemini)

### Setup

```bash
git clone <url-do-repositorio>
cd agente-trabalhista-langchain-chromadb-gemini

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env
# Edite .env e adicione: GOOGLE_API_KEY=sua_chave_aqui
```

### 1. Indexar a legislação

```bash
python indexar.py
```

Na primeira execução, baixa e indexa todas as 7 fontes. Nas execuções seguintes, oferece menu interativo para indexação incremental.

### 2. Iniciar o assistente

```bash
streamlit run app.py
```

Acesse `http://localhost:8501`, preencha seu perfil na barra lateral e faça sua pergunta.

---

## Estrutura do Projeto

```
.
├── app.py           # Interface Streamlit + pipeline de consulta RAG
├── indexar.py       # Pipeline de indexação: scraping → chunking → embedding → ChromaDB
├── requirements.txt
├── .env.example
└── chroma_rh/       # Banco vetorial persistido (gerado pelo indexar.py)
    ├── registro.json
    └── <uuid>/      # Arquivos internos do ChromaDB (HNSW index)
```

---

## Fluxo Detalhado de uma Consulta

```
1. Usuário preenche perfil (polo + tipo de vínculo)
2. Usuário digita a pergunta
3. Gemini Embedding converte a pergunta em vetor
4. ChromaDB faz similarity_search filtrando por tipo_trabalhador (top-8)
5. Gemini 2.5 Flash reordena os 8 trechos por relevância (score 0-10)
6. Top-5 trechos rerankeados compõem o contexto jurídico
7. Gemini gera resposta personalizada com citações de artigos
8. Streamlit exibe resposta + fontes consultadas em expander
9. Mensagem é salva no histórico da sessão
```

---

## Variáveis de Ambiente

| Variável | Descrição |
|---|---|
| `GOOGLE_API_KEY` | Chave da API Google AI Studio (Gemini) |

---

## Aviso Legal

Este projeto é uma ferramenta de **auxílio informativo** e **não substitui consultoria jurídica profissional**. Toda resposta gerada pelo agente inclui o aviso de que se trata de uma IA sem formação em direito. Para questões legais concretas, consulte um advogado trabalhista.
