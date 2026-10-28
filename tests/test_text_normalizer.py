from src.utils.text_normalizer import basic_normalize, normalize_text 

def test_basic_normalize():
    """Prueba la normalización básica de espacios y minúsculas."""
    assert basic_normalize("  Hola   Món  ") == "hola món"
    assert basic_normalize("TEXT\nAMB\tSALTS") == "text amb salts"
    assert basic_normalize(None) == "none"

def test_normalize_text_basic():
    """Prueba la normalización estándar de un texto con signos de puntuación."""
    text = "Hola, com estem? Això és una prova."
    assert normalize_text(text) == "hola com estem això és una prova"

def test_normalize_text_tags_and_noise():
    """Prueba la eliminación de etiquetas de transcripción y ruido."""
    text = "Hola [risas] que tal ((ruido)) <unk> {pausa}"
    assert normalize_text(text) == "hola que tal"

def test_normalize_text_geminada():
    """Prueba la corrección de la 'ela geminada' con punto o punto volat."""
    assert normalize_text("col.lecció") == "col·lecció"
    assert normalize_text("col·lecció") == "col·lecció"

def test_normalize_text_punctuation_and_hyphens():
    """Prueba que los apóstrofos y guiones se conviertan en espacios."""
    text = "L'avi de l'horta - ràpidament-va dir"
    assert normalize_text(text) == "l avi de l horta ràpidament va dir"

def test_normalize_text_strip_accents():
    """Prueba la eliminación de tildes (acentos gráficos) cuando se activa el flag."""
    text = "Avi, càtering iústria"
    assert normalize_text(text, strip_accents=True) == "avi catering iustria"

def test_normalize_text_edge_cases():
    """Prueba casos límite como cadenas vacías o valores nulos."""
    assert normalize_text(None) == ""
    assert normalize_text("") == ""
    assert normalize_text("   ") == ""