# DocRouter

Route a document through the cheapest model that is confident enough.

Parsers cover PDF, email, and scanned images. Classification tries TF-IDF first, then DistilBERT, then Gemini.

## Run

```bash
uvicorn src.api.main:app --reload
```

POST a file to /process-document. Set GEMINI_API_KEY in a .env file at the repo root when you want the LLM tier.