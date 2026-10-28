from src.utils.text_normalizer import basic_normalize, normalize_text 

def test_basic_normalize():
    assert basic_normalize("  Hola   Món  ") == "hola món"
    assert basic_normalize("TEXT\nAMB\tSALTS") == "text amb salts"
    assert basic_normalize(None) == "none"

def test_normalize_text_basic():
    text = "Hola, com estem? Això és una prova."
    assert normalize_text(text) == "hola com estem això és una prova"

def test_normalize_text_tags_and_noise():
    text = "Hola [risas] que tal ((ruido)) <unk> {pausa}"
    assert normalize_text(text) == "hola que tal"

def test_normalize_text_geminada():
    assert normalize_text("col.lecció") == "col·lecció"
    assert normalize_text("col·lecció") == "col·lecció"

def test_normalize_text_punctuation_and_hyphens():
    text = "L'avi de l'horta - ràpidament-va dir"
    assert normalize_text(text) == "l avi de l horta ràpidament va dir"

def test_normalize_text_strip_accents():
    text = "Avi, càtering iústria"
    assert normalize_text(text, strip_accents=True) == "avi catering iustria"

def test_normalize_text_acronyms():
    """Valida la limpieza de siglas puntuadas."""
    assert normalize_text("La o.n.u. ho ha confirmat") == "la onu ho ha confirmat"

def test_normalize_text_truncated():
    """Valida que los truncamientos de transcripción se procesen según el flag."""
    text = "estava pa- pa- parlant"
    assert normalize_text(text, drop_truncated=True) == "estava parlant"
    assert normalize_text(text, drop_truncated=False) == "estava pa pa parlant"

def test_normalize_text_numbers_and_ordinals():
    """Valida la conversión a texto de cardinales, decimales y ordinales."""
    text = "L'any 2009 el 1r guanyador va rebre 3,5 punts"
    result = normalize_text(text, numbers_to_words=True, lang="ca")
    assert "dos mil nou" in result
    assert "primer" in result
    assert "tres coma cinc" in result

def test_normalize_text_fillers():
    """Valida la eliminación opcional de muletillas."""
    text = "eh mmm hola què tal uh ahh"
    assert normalize_text(text, drop_fillers=True) == "hola què tal"
    assert normalize_text(text, drop_fillers=False) == "eh mmm hola què tal uh ahh"

def test_normalize_text_edge_cases():
    assert normalize_text(None) == ""
    assert normalize_text("") == ""
    assert normalize_text("   ") == ""