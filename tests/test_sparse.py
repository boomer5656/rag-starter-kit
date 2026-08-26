from __future__ import annotations

from ragkit.sparse import (
    CorpusStats, doc_sparse, query_sparse, term_index, to_qdrant, tokenize,
)


def test_tokenize_lowercases_filters_short_and_punct():
    assert tokenize("The Cap-Rate is 12% on Unit A!") == ["the", "cap", "rate", "is", "12", "on", "unit"]
    # single-char tokens ("a") and punctuation dropped; digits kept.


def test_term_index_stable_and_u32():
    a, b = term_index("balance"), term_index("balance")
    assert a == b and 0 <= a < 2**32
    assert term_index("balance") != term_index("escrow")


def test_add_doc_counts_df_once_per_doc():
    s = CorpusStats()
    s.add_doc("rent rent rent")          # 3 tokens, but df for 'rent' is +1
    s.add_doc("rent escrow")
    assert s.n_docs == 2
    assert s.total_len == 5
    assert s.df[term_index("rent")] == 2
    assert s.df[term_index("escrow")] == 1
    assert s.avgdl == 2.5


def test_idf_rarer_term_scores_higher():
    s = CorpusStats()
    for _ in range(9):
        s.add_doc("common")
    s.add_doc("common rare")
    assert s.idf(term_index("rare")) > s.idf(term_index("common"))


def test_doc_sparse_saturates_with_tf():
    s = CorpusStats()
    s.add_doc("aa bb cc dd")   # avgdl context (tokens must be >=2 chars)
    one = doc_sparse("zz", s)[term_index("zz")]
    many = doc_sparse("zz zz zz zz zz zz", s)[term_index("zz")]
    assert many > one              # more occurrences -> higher weight
    assert many < 6 * one          # ...but saturates (not linear)


def test_query_sparse_is_idf_valued():
    s = CorpusStats()
    s.add_doc("alpha beta")
    q = query_sparse("alpha alpha", s)     # dedup to unique terms
    assert list(q.keys()) == [term_index("alpha")]
    assert q[term_index("alpha")] == s.idf(term_index("alpha"))


def test_bm25_dot_rewards_matching_rare_term():
    s = CorpusStats()
    for _ in range(20):
        s.add_doc("filler text here")
    s.add_doc("the escrow balance rose")
    qv = query_sparse("escrow", s)
    dv_match = doc_sparse("the escrow balance rose", s)
    dv_miss = doc_sparse("filler text here", s)
    dot = lambda q, d: sum(v * d.get(k, 0.0) for k, v in q.items())
    assert dot(qv, dv_match) > 0
    assert dot(qv, dv_miss) == 0


def test_to_qdrant_shape_and_empty():
    out = to_qdrant({5: 0.5, 9: 0.25})
    assert set(out.keys()) == {"indices", "values"}
    assert len(out["indices"]) == len(out["values"]) == 2
    assert to_qdrant({}) == {"indices": [], "values": []}


def test_corpus_stats_roundtrip(tmp_path):
    s = CorpusStats()
    s.add_doc("alpha beta gamma")
    s.add_doc("alpha delta")
    p = str(tmp_path / "stats.json")
    s.save(p)
    r = CorpusStats.load(p)
    assert r.n_docs == s.n_docs and r.total_len == s.total_len and r.df == s.df
    assert CorpusStats.load(str(tmp_path / "missing.json")).n_docs == 0


def test_stats_path_shape():
    from ragkit.sparse import stats_path
    assert stats_path("mycoll").replace("\\", "/") == ".ragkit/sparse/mycoll.json"
