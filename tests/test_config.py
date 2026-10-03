def test_config_variables():
    from app import config
    assert config.DB_NAME == "hybrid_rag"
    assert config.VECTOR_INDEX == "chunk_vector_index"
    assert config.MIN_TABLE_ROWS == 8
    assert config.EMBED_DIM == 768
    assert bool(config.MONGODB_URI) is True
    assert bool(config.GEMINI_API_KEY) is True
