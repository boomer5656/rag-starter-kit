# Idempotency and the dimension-mismatch guard

Re-running an ingest over the same folder should never create duplicate
chunks. ragkit achieves this structurally rather than by checking for
existing entries first: every chunk's point id in Qdrant is a deterministic
hash of its source URI and its ordinal position within that source. Ingest
the same file again and the same chunk lands on the same point id, so
Qdrant's upsert replaces it in place instead of adding a new point next to
it. Re-chunking a changed document clears its old points before writing the
new ones, so a document that used to produce eight chunks and now produces
six doesn't leave two orphaned points behind.

The state store follows the same pattern: it's a SQLite (or optionally
Postgres) table keyed by collection and source URI, updated with an
`ON CONFLICT DO UPDATE` on every stage transition. Re-running ingest is
always safe.

A second correctness guarantee is around embedding dimensions. Every
embedding model produces vectors of a fixed size — bge-m3 produces 1024
dimensions, other models produce other sizes. If a collection was created
with one model's dimension and you later point ragkit at a different model,
writing mismatched vectors into the same collection would silently corrupt
search results. ragkit checks the embedder's actual output dimension against
the collection's configured dimension before any write happens. If they
don't match, ingestion refuses to proceed and reports the mismatch loudly,
rather than letting a 768-dimension query wander into a 1024-dimension
collection (or vice versa).
