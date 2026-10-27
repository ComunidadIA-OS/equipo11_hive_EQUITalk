import re
import unicodedata

# Marcas de Transcriber / ruido: [risas], (( )), <unk>, {pausa}
_TAGS_RE = re.compile(r"\[[^\]]*\]|\(\([^)]*\)\)|\([^)]*\)|<[^>]*>|\{[^}]*\}")
# ela geminada escrita con punto ("col.lecció") -> "col·lecció"
_GEMINADA_RE = re.compile(r"(?<=[lL])[.•](?=[lL])"w)
# todo lo que no sea letra/número/espacio/punt volat
_PUNCT_RE = re.compile(r"[^\w\s·]|_", flags=re.UNICODE)
_SPACES_RE = re.compile(r"\s+")


def basic_normalize(text: str) -> str:
    """Normalización de minúsculas y espacios."""
    return " ".join(str(text).lower().split())


def normalize_text(text: str, strip_accents: bool = False) -> str:
    """Minúsculas, sin marcas ni puntuación, apóstrofos/guiones como espacio.
    """
    if text is None:
        return ""
    text = unicodedata.normalize("NFC", str(text)).lower()
    text = _TAGS_RE.sub(" ", text)
    text = text.replace("’", "'").replace("‘", "'").replace("`", "'")
    text = _GEMINADA_RE.sub("·", text)
    text = text.replace("'", " ").replace("-", " ").replace("–", " ").replace("—", " ")
    text = _PUNCT_RE.sub(" ", text)
    if strip_accents:
        text = "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")
        text = unicodedata.normalize("NFC", text)
    return _SPACES_RE.sub(" ", text).strip()