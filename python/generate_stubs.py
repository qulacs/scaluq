import os
import sys

generated_stub = sys.argv[1]
install_path = sys.argv[2]

INST_REPLACE_SIGNATURE_TO = '[replace signature to] '

transformed_lines = []
with open(generated_stub, 'r') as f:
    original_lines = f.readlines()

i = 0
while i < len(original_lines):
    first_line = original_lines[i].strip()
    if first_line.startswith('from') or first_line.startswith('import') or first_line == '':
        transformed_lines.append(original_lines[i])
        i += 1
        continue
    if first_line.startswith('@'):
        # decorator is stored as is
        transformed_lines.append(original_lines[i])
        i += 1
    assert i < len(original_lines)
    indent = len(original_lines[i]) - len(original_lines[i].lstrip())
    signature = original_lines[i].strip()
    if i + 1 == len(original_lines) or not (original_lines[i + 1].strip().startswith('"""') or original_lines[i + 1].strip().startswith(r'r"""')):
        # no docstring
        transformed_lines.append(original_lines[i])
        i += 1
        continue
    doc_begin = i + 1
    if original_lines[doc_begin].strip() not in ['"""', r'r"""']:
        assert original_lines[doc_begin].strip().endswith('"""')
        content = original_lines[doc_begin].strip()[3:-3]
        if content.startswith(INST_REPLACE_SIGNATURE_TO):
            signature = content[len(INST_REPLACE_SIGNATURE_TO):] + ':'
            transformed_lines.append(' ' * indent + signature + '\n')
        else:
            transformed_lines.append(' ' * indent + signature + '\n')
            transformed_lines.append(original_lines[doc_begin])
        i = doc_begin + 1
    else:
        doc_end = doc_begin + 2
        while doc_end < len(original_lines) and '"""' not in original_lines[doc_end].strip():
            doc_end += 1
        assert doc_end < len(original_lines)
        assert original_lines[doc_end].strip() == '"""'
        transformed_doc = []
        for j in range(doc_begin + 1, doc_end):
            line = original_lines[j].strip()
            if line.startswith(INST_REPLACE_SIGNATURE_TO):
                signature = line[len(INST_REPLACE_SIGNATURE_TO):] + ':'
            else:
                transformed_doc.append(original_lines[j])
        transformed_lines.append(' ' * indent + signature + '\n')
        transformed_lines.append(original_lines[doc_begin])
        transformed_lines += transformed_doc
        transformed_lines.append(original_lines[doc_end])
        i = doc_end + 1

with open(os.path.join(install_path, '__init__.pyi'), 'a') as f:
    f.writelines(transformed_lines)
with open(os.path.join(install_path, 'py.typed'), 'w') as f:
    pass
