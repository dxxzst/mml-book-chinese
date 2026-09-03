import re
import os
import sys

def check_file(path):
    print(f'Checking {path}...')
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    errors = 0
    warnings = 0

    # 1. Check double dollar tags
    blocks = re.findall(r'\$\$(.*?)\$\$', content, re.DOTALL)
    for i, b in enumerate(blocks):
        tags = re.findall(r'\\tag\{.*?\}', b)
        if len(tags) > 1:
            print(f'  [ERROR] Multiple \\tag found in block {i}: {tags}')
            errors += 1
        if not b.strip():
            print(f'  [ERROR] Empty $$ block found at index {i}')
            errors += 1

    # 2. Check underscores touching outside Chinese without spaces: e.g. "中文_term_" or "_term_中文"
    # An underscore delimiter should have whitespace or punctuation before the opening '_' and after the closing '_'
    bad_lead = re.findall(r'[\u4e00-\u9fa5]_[^\s_]', content)
    bad_trail = re.findall(r'[^\s_]_[\u4e00-\u9fa5]', content)
    if bad_lead or bad_trail:
        print(f'  [WARNING] Underscore touching CJK without spacing: {bad_lead + bad_trail}')
        warnings += 1

    # 3. Check math inside HTML
    html_math = re.findall(r'<[a-z]+[^>]*>.*?\$.*?</[a-z]+>', content)
    if html_math:
        print(f'  [ERROR] Math in HTML tag: {html_math[:3]}')
        errors += 1
        
    # 4. Check image links
    imgs = re.findall(r'<img\s+src=[\"\'](.*?)[\"\']', content)
    for img in imgs:
        full = os.path.normpath(os.path.join(os.path.dirname(path), img))
        if not os.path.exists(full):
            print(f'  [ERROR] Missing image file: {img} -> {full}')
            errors += 1
        else:
            print(f'  [OK] Image exists: {img}')
            
    print(f'Done checking {path}: {errors} errors, {warnings} warnings.\n')
    return errors == 0 and warnings == 0

if __name__ == '__main__':
    for p in sys.argv[1:]:
        check_file(p)
