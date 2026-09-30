"""Rebuild the bibliography from the original printed-book PDF.

Usage: python scripts/build_references.py /path/to/mml-book_printed.pdf
Use --output to review generated Markdown without replacing 3.References.md.
"""

import argparse
from pathlib import Path
import re

# Accent map
accents = {
    '´e': 'é', '´E': 'É', '`e': 'è', '`a': 'à', '´a': 'á', '´o': 'ó', '¨o': 'ö', '¨u': 'ü',
    '¨a': 'ä', '´ı': 'í', '´i': 'í', '˘c': 'č', 'ˇs': 'š', 'ˇz': 'ž', '˜n': 'ñ', '´c': 'ć',
    'ø': 'ø', '´Equations': 'Équations', 'G´en´erales': 'Générales', 'Quatri`eme': 'Quatrième',
    'D´emonstration': 'Démonstration', 'Impossibilit´e': 'Impossibilité', 'R´esolution': 'Résolution',
    'Alg´ebrique': 'Algébrique', 'Degr´e': 'Degré', 'Daum´e': 'Daumé', 'Sch¨olkopf': 'Schölkopf',
    'R´enyi': 'Rényi', 'Kullback–Leibler': 'Kullback-Leibler', 'Fr´echet': 'Fréchet',
    'Poincar´e': 'Poincaré', 'G¨artner': 'Gärtner', 'Dost´al': 'Dostál', 'Kre˘ın': 'Kreĭn',
    'Krishnapuram': 'Krishnapuram', 'F´evotte': 'Févotte'
}

def clean_text(t):
    for k, v in accents.items():
        t = t.replace(k, v)
    t = re.sub(r'\s+', ' ', t).strip()
    return t

# Only repair a physical line break in a hyphenated URL hostname. Requiring
# http(s)://, no preceding path, and a domain-shaped continuation avoids changing
# prose such as "Op-\ntimization" or attaching prose after a complete URL.
WRAPPED_HOSTNAME = re.compile(
    r"(https?://[A-Za-z0-9.-]*-)[ \t]*\r?\n[ \t]*"
    r"(?=[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+(?:[/:?#]|[.,]?(?:\s|$)))"
)
URL = re.compile(r"https?://[^\s<>\[\]]+")


def markdown_url(match):
    """Keep bibliography punctuation outside explicit Markdown link targets."""
    token = match.group()
    url = token.rstrip(".,;:!?")
    # A closing parenthesis can belong to the URL or to the surrounding prose.
    while url.endswith(")") and url.count(")") > url.count("("):
        url = url[:-1]
    suffix = token[len(url):]
    # Angle-bracket destinations also support balanced parentheses in URLs.
    return f"[{url}](<{url}>){suffix}"


def format_reference(lines):
    """Join PDF lines without inserting whitespace into wrapped hostnames."""
    text = WRAPPED_HOSTNAME.sub(r"\1", "\n".join(lines))
    return URL.sub(markdown_url, clean_text(text))


def extract_references(doc):
    """Extract the original bibliography pages, retaining PDF line boundaries."""
    entries = []
    current_entry = []
    for pno in range(400, 412):
        page = doc[pno]
        blocks = page.get_text('blocks')
        for b in blocks:
            text = b[4].strip()
            lines = text.split('\n')
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                if line in ['References', 'Classiﬁcation with Support Vector Machines'] or re.match(r'^\d+$', line):
                    continue
                if 'Draft (' in line or 'To be published by Cambridge' in line or 'Mathematics for Machine Learning' in line or 'https://mml-book.com' in line or 'c⃝' in line:
                    continue
                if re.match(r"^[A-Z][a-zA-Z\s\.\,\-\'\`\´\(\)]+, [A-Z].*\b(19\d\d|20\d\d|18\d\d|17\d\d)\b", line):
                    if current_entry:
                        entries.append(format_reference(current_entry))
                        current_entry = []
                current_entry.append(line)
    if current_entry:
        entries.append(format_reference(current_entry))
    return entries


def write_references(entries, output):
    with Path(output).open('w', encoding='utf-8') as f:
        f.write('# 参考文献 (References)\n\n')
        f.write('本书涉及的所有学术专著、经典论文及技术报告的完整参考文献列表如下：\n\n')
        for i, entry in enumerate(entries, 1):
            f.write(f'{i}. {entry}\n\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pdf', type=Path, help='Original mml-book_printed.pdf')
    parser.add_argument('--output', type=Path, default=Path('3.References.md'))
    args = parser.parse_args()
    # Tests for text joining do not need PyMuPDF or a local copy of the PDF.
    import pymupdf

    with pymupdf.open(args.pdf) as doc:
        entries = extract_references(doc)
    write_references(entries, args.output)
    print(f'Wrote {len(entries)} references to {args.output}')


if __name__ == '__main__':
    main()
