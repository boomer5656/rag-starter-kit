# Connectors

A connector's job is small: yield a `SourceDoc` for each item it knows
about. A `SourceDoc` carries a stable URI, a title, a MIME type, and either
already-extracted text or raw bytes for the pipeline to extract later (via
Tika, for binary formats like PDF and docx). Everything else — extraction,
gating, chunking, embedding, storage, and state tracking — is handled by
the pipeline, so a new connector is usually a short piece of code.

The headline connector walks a local directory and yields one `SourceDoc`
per file, which covers the common case of "I have a folder of PDFs, docx
files, and markdown notes I want searchable." Plain text and markdown files
are read directly; everything else is handed to the Tika extraction sidecar,
which is confined to an allowed root directory so a connector can't be
tricked into reading files outside the folder you pointed it at.

Beyond the file-directory case, a connector can pull from anywhere: an API,
a database export, a message queue. As long as it can produce a URI that's
stable across runs (so re-ingesting updates the same document rather than
duplicating it) and either text or bytes, it plugs into the same pipeline
with the same observability, idempotency, and dimension guarantees as the
built-in connector.
