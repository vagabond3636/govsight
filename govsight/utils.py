import re
import unicodedata

def slugify(value: str) -> str:
    """
    Converts strings to lowercase, removes non-ASCII characters,
    replaces spaces and punctuation with hyphens, and strips leading/trailing hyphens.
    """
    value = str(value)
    value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode('ascii')
    value = re.sub(r'[^\w\s-]', '', value).strip().lower()
    return re.sub(r'[-\s]+', '-', value)
