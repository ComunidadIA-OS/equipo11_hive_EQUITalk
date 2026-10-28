import logging
import re
import unicodedata

logger = logging.getLogger(__name__)

from num2words import num2words as _num2words

_WARNED_NUM2WORDS = False

# Marcas de Transcriber / ruido: [risas], (( )), <unk>, {pausa}
_TAGS_RE = re.compile(r"\[[^\]]*\]|\(\([^)]*\)\)|\([^)]*\)|<[^>]*>|\{[^}]*\}")
# ela geminada escrita con punto o viñeta ("col.lecció") -> "col·lecció"
_GEMINADA_RE = re.compile(r"(?<=[lL])[.•∙⋅](?=[lL])")
# siglas con puntos: "o.n.u." -> "onu"
_ACRONYM_RE = re.compile(r"\b(?:[^\W\d_]\.){2,}")
# palabras truncadas en la referencia: "pa- pa- parlant" (guion final, sin letra detrás)
_TRUNCATED_RE = re.compile(r"(?<![\w-])[^\W\d_]+-(?![\w-])")
# ordinales pegados al número: 1r, 2n, 3r, 4t, 5è, 1a, 1º
_ORDINAL_RE = re.compile(r"(?<!\w)(\d+)(?:r|n|t|è|er|a|º|ª)(?!\w)")
# números sueltos (no pegados a letras): 2009, 1.500, 3,5
_NUM_RE = re.compile(r"(?<!\w)\d+(?:[.,]\d+)*(?!\w)")
# todo lo que no sea letra/número/espacio/punt volat
_PUNCT_RE = re.compile(r"[^\w\s·]|_", flags=re.UNICODE)
# punt volat que no está entre dos eles
_ORPHAN_DOT_RE = re.compile(r"(?<![lL])·|·(?![lL])")
_SPACES_RE = re.compile(r"\s+")

# Interjecciones que Whisper casi nunca transcribe y que la referencia sí suele llevar.
# (No incluye "em", "en", "a"... que son palabras reales en catalán.)
_FILLERS = frozenset({
    "eh", "ehh", "eeh", "mm", "mmm", "mmmm", "hm", "hmm", "uh", "uhm", "um",
    "ah", "ahh", "oh", "ohh",
})


def basic_normalize(text: str) -> str:
    """Normalización de minúsculas y espacios."""
    return " ".join(str(text).lower().split())


def _n2w(n: int, lang: str, ordinal: bool = False) -> str:
    if ordinal:
        try:
            return _num2words(n, lang=lang, to="ordinal")
        except Exception:
            pass  # el idioma no soporta ordinales: se usa el cardinal
    return _num2words(n, lang=lang)


def _num_to_words(tok: str, lang: str) -> str:
    """'2009' -> 'dos mil nou'; '1.500' -> 'mil cinc-cents'; '3,05' -> 'tres coma zero cinc'."""
    try:
        if re.fullmatch(r"\d{1,3}(?:\.\d{3})+(?:,\d+)?", tok):       # separador de millares
            ip, _, dp = tok.replace(".", "").partition(",")
        elif re.fullmatch(r"\d+[.,]\d+", tok):                       # decimal
            ip, dp = re.split(r"[.,]", tok)
        elif tok.isdigit():
            ip, dp = tok, ""
        else:                                                         # '1,2,3' u otros: uno a uno
            return " ".join(_num_to_words(t, lang) for t in re.split(r"[.,]", tok) if t)
        words = _n2w(int(ip), lang)
        if dp:
            zero = "cero" if lang == "es" else "zero"
            lead, rest = len(dp) - len(dp.lstrip("0")), dp.lstrip("0")
            frac = [zero] * lead + ([_n2w(int(rest), lang)] if rest else [])
            words += " coma " + " ".join(frac)
        return words
    except Exception:
        return tok


def _numbers_to_words(text: str, lang: str) -> str:
    global _WARNED_NUM2WORDS
    if lang == "ca":
        text = text.replace("%", " per cent ")
    elif lang == "es":
        text = text.replace("%", " por ciento ")
    text = text.replace("€", " euros ")
    if _num2words is None:
        if _NUM_RE.search(text) and not _WARNED_NUM2WORDS:
            logger.warning("num2words no está instalado: los números se dejan como dígitos "
                           "(pip install num2words).")
            _WARNED_NUM2WORDS = True
        return text
    text = _ORDINAL_RE.sub(lambda m: " " + _safe_ordinal(int(m.group(1)), lang) + " ", text)
    return _NUM_RE.sub(lambda m: " " + _num_to_words(m.group(0), lang) + " ", text)


def _safe_ordinal(n: int, lang: str) -> str:
    try:
        return _n2w(n, lang, ordinal=True)
    except Exception:
        return str(n)


def normalize_text(text: str, strip_accents: bool = False, lang: str = "ca",
                   numbers_to_words: bool = True, drop_fillers: bool = True,
                   drop_truncated: bool = True) -> str:
    """
    Minúsculas, sin marcas ni puntuación, apóstrofos/guiones como espacio.
    """
    if text is None:
        return ""
    lang = (lang or "ca").lower()[:2]
    text = unicodedata.normalize("NFC", str(text)).lower()
    text = _TAGS_RE.sub(" ", text)
    text = text.replace("’", "'").replace("‘", "'").replace("`", "'").replace("´", "'")
    text = text.replace("ŀ", "l·")                      # U+0140 (l con punt volat precompuesta)
    text = _GEMINADA_RE.sub("·", text)
    text = _ACRONYM_RE.sub(lambda m: m.group(0).replace(".", ""), text)
    if numbers_to_words:
        text = _numbers_to_words(text, lang)
    if drop_truncated:
        text = _TRUNCATED_RE.sub(" ", text)
    text = text.replace("'", " ").replace("-", " ").replace("–", " ").replace("—", " ")
    text = _PUNCT_RE.sub(" ", text)
    text = _ORPHAN_DOT_RE.sub(" ", text)
    if strip_accents:
        text = "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")
        text = unicodedata.normalize("NFC", text)
    tokens = _SPACES_RE.sub(" ", text).strip().split()
    if drop_fillers:
        tokens = [t for t in tokens if t not in _FILLERS]
    return " ".join(tokens)