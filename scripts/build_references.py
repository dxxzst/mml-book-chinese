import pymupdf
import re

doc = pymupdf.open(r'E:\需求记录\文档资料\Books\mml-book_printed.pdf')

entries = []
current_entry = []

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

for pno in range(400, 412):
    page = doc[pno]
    blocks = page.get_text('blocks')
    for b in blocks:
        text = b[4].strip()
        lines = text.split('\n')
        for l in lines:
            l = l.strip()
            if not l:
                continue
            if l in ['References', 'Classiﬁcation with Support Vector Machines'] or re.match(r'^\d+$', l):
                continue
            if 'Draft (' in l or 'To be published by Cambridge' in l or 'Mathematics for Machine Learning' in l or 'https://mml-book.com' in l or 'c⃝' in l:
                continue
            if re.match(r'^[A-Z][a-zA-Z\s\.\,\-\'\`\´\(\)]+, [A-Z].*\b(19\d\d|20\d\d|18\d\d|17\d\d)\b', l):
                if current_entry:
                    entries.append(clean_text(' '.join(current_entry)))
                    current_entry = []
            current_entry.append(l)

if current_entry:
    entries.append(clean_text(' '.join(current_entry)))

with open('3.References.md', 'w', encoding='utf-8') as f:
    f.write('# 参考文献 (References)\n\n')
    f.write('本书涉及的所有学术专著、经典论文及技术报告的完整参考文献列表如下：\n\n')
    for i, e in enumerate(entries, 1):
        f.write(f'{i}. {e}\n\n')

print(f'Wrote {len(entries)} references to 3.References.md')
