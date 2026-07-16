# What ragkit is

ragkit is a self-hosted pipeline for getting your own documents into a vector
store so you can search and chat over them. Retrieval-augmented generation
(RAG) chat UIs are everywhere now — Open WebUI, Dify, AnythingLLM all ship
one for free. What they mostly treat as a black box is the unglamorous part:
turning a folder of messy real-world files into well-chunked, correctly
embedded, queryable data. That's the part ragkit focuses on.

Everything runs on hardware you control. Embeddings come from your own
Ollama instance, vectors live in your own Qdrant, and nothing is sent to a
third-party API unless you explicitly wire one in. For documents that can't
leave your network — client files, medical records, anything under NDA — that
matters more than the chat UI on top of it.

ragkit is not a chat application and not a plugin framework. It is a small
Python package plus a couple of optional sidecars (Tika for extraction,
a reranker for better result ordering) that you point at a directory and
run. The output is a Qdrant collection you can query yourself, search from
the `ragkit` CLI, or expose to an MCP client like Claude Code.
