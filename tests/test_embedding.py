from backend.embedding import embed_text

def test_embed_text():
    test = "Test text"
    emb = embed_text(test)
    assert isinstance(emb, list)
    assert len(emb) > 0