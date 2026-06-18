# Rag-chatbot

Open-source **Retrieval-Augmented Generation** for company policies and legal documents.

**Main project:** [`company_policy_rag/`](company_policy_rag/) — production-minded local RAG with LlamaIndex, Ollama, ChromaDB, Streamlit, Docker, and a golden-set eval harness.

## Quick links

| Resource | Description |
|----------|-------------|
| [company_policy_rag/README.md](company_policy_rag/README.md) | Full documentation, quick start, architecture |
| [company_policy_rag/README2.md](company_policy_rag/README2.md) | Engineering build story and metric regressions |
| [company_policy_rag/docs/](company_policy_rag/docs/) | Interview notes, engineering plans |

## Highlights

- Grounded answers with mandatory `[Source N]` citations
- Over-retrieve → rerank → score filter retrieval pipeline
- Strict / balanced faithfulness modes + optional guard
- 15-case golden eval with trend logging (`logs/evaluation_results.json`)
- Streamlit chat UI with **legal PDF upload** from the browser
- Docker Compose + host Ollama deployment

## Quick start

```bash
git clone https://github.com/SoubhagyaJain/Rag-chatbot.git
cd Rag-chatbot/company_policy_rag

python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env

ollama pull qwen2.5:7b
ollama pull nomic-embed-text

python scripts/index_documents.py
streamlit run app/streamlit_app.py
```

Docker (build locally):

```bash
cd company_policy_rag
cp .env.docker.example .env.docker
docker compose up --build
```

Docker (pull pre-built image from Docker Hub):

```bash
git clone https://github.com/SoubhagyaJain/Rag-chatbot.git
cd Rag-chatbot/company_policy_rag
cp .env.docker.example .env.docker
docker pull soubhagyajain/rag-chatbot:latest
docker compose -f docker-compose.dockerhub.yml up -d
```

> **Note:** GitHub Packages (ghcr.io) is disabled on this account until GitHub Support re-enables it. Use Docker Hub above, or build locally with `docker compose up --build`.

PyPI (library + CLI — run from project directory with `data/`, `.env`):

```bash
pip install soubhagya-policy-rag
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
cd company_policy_rag && cp .env.example .env
policy-rag-index
policy-rag-chat
```

Open [http://localhost:8501](http://localhost:8501).

## Measured quality (golden set, balanced mode)

| Metric | Best run |
|--------|----------|
| Answer Relevancy | 0.747 |
| Context Precision | 0.80 |
| Faithfulness | 0.807 |
| Hit Rate | 0.867 |

See [evaluation section](company_policy_rag/README.md#evaluation) for how to reproduce.

## License

MIT — see [LICENSE](LICENSE). Sample handbook PDF is for demo indexing only; replace with your own documents in production.