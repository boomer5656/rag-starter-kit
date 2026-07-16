# The pipeline stages

Every document ragkit ingests moves through a fixed sequence of explicit
states: `discovered -> extracted -> gated -> chunked -> embedded -> stored`.
A document can also land in `skipped` (the relevance gate rejected it on
purpose) or `error` (something failed). Nothing is ever silently dropped —
`ragkit status` can always tell you exactly what happened to a given file.

The stages run in batches rather than one document at a time. On a
single-GPU homelab box, loading and unloading a model per document is
expensive — swapping the embedding model in and out can cost tens of
seconds per file. So ragkit processes a whole batch through extraction,
then a whole batch through the optional gate, then a whole batch through
chunking, and only then embeds and stores everything together. Each model
stays resident for the whole stage instead of thrashing.

The relevance gate is optional and off by default. When enabled, a small,
cheap model looks at each document before the expensive embedding step and
decides whether it's worth keeping. This is useful when a source directory
contains a lot of noise — boilerplate, duplicates, irrelevant attachments —
and you'd rather not spend embedding time and vector storage on it.

An error at any stage is isolated to that one document. The rest of the
batch keeps moving; the failed document is marked `error` with a message
and stays visible in the state store rather than disappearing.
