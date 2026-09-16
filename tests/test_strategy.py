from rag_system.models import RetrievalStrategy

def test_from_value_defaults_to_hybrid_for_invalid_values():
    assert RetrievalStrategy.from_value("unknown",default=RetrievalStrategy.DECOMPOSED) is RetrievalStrategy.DECOMPOSED
    assert RetrievalStrategy.from_value(None) is RetrievalStrategy.DECOMPOSED


def test_from_value_parses_valid_value():
    assert RetrievalStrategy.from_value("semantic") is RetrievalStrategy.SEMANTIC


