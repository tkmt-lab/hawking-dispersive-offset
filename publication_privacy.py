"""Bounded offline privacy-pattern scan; no network or mathematical verification."""
from pathlib import Path
import hashlib
import ast
import json
import re
import shutil
import subprocess
import sys
import zlib

# Byte-encoded match vocabulary prevents a scanner's own source from matching it.
TOKENS = {
    'local-volume-root': '2f566f6c756d65732f',
    'local-user-root': '2f55736572732f',
    'private-root': '2f707269766174652f',
    'temporary-root': '2f746d702f',
    'os-temp-root': '2f7661722f666f6c646572732f',
    'home-root': '2f686f6d652f',
}
PUBLIC_PERSON = {'given-names': 'Tatsuhiko', 'family-names': 'Takimoto'}
PATTERNS = [(key, re.compile(re.escape(bytes.fromhex(value).decode()), re.I))
            for key, value in TOKENS.items()]
PATTERNS += [
    ('email-address', re.compile(r'\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b', re.I)),
    ('private-hostname', re.compile(r'\b[a-z0-9][a-z0-9.-]*\.(?:local|internal|lan|ts\.net)\b', re.I)),
    ('private-ipv4', re.compile(r'\b(?:10(?:\.\d{1,3}){3}|192\.168(?:\.\d{1,3}){2}|172\.(?:1[6-9]|2\d|3[01])(?:\.\d{1,3}){2})\b')),
    ('windows-user-root', re.compile(r'[A-Z]:[\\/]Users[\\/]', re.I)),
]


def mask_public_identity(text):
    # The approved public author's name is not a machine login. Limit exceptions
    # to the full name and the exact given-name field used by the citation object.
    given = PUBLIC_PERSON['given-names']
    text = re.sub(re.escape(given) + r'\s+' + re.escape(PUBLIC_PERSON['family-names']), '<PUBLIC_AUTHOR>', text, flags=re.I)
    return re.sub(r'''(["']given-names["']\s*:\s*["'])''' + re.escape(given) + r'''(["'])''',
                  r'\1<PUBLIC_GIVEN_NAME>\2', text, flags=re.I)


def matches(text):
    text = mask_public_identity(text)
    found = {key for key, pattern in PATTERNS if pattern.search(text)}
    return sorted(found)


def file_views(path):
    raw = path.read_bytes()
    yield 'raw-utf8', raw.decode('utf-8', errors='replace')
    if raw.startswith((b'\xff\xfe', b'\xfe\xff')) or b'\x00' in raw:
        for enc in ('utf-16-le', 'utf-16-be'):
            yield enc, raw.decode(enc, errors='replace')
    if path.suffix.lower() == '.pdf':
        for tool, args in [('pdftotext', [str(path), '-']), ('pdfinfo', [str(path)])]:
            assert shutil.which(tool), 'privacy scan requires existing Poppler utility: ' + tool
            proc = subprocess.run([tool, *args], capture_output=True, timeout=30)
            assert proc.returncode == 0, ('PDF extraction failed', path.name, tool)
            yield tool, proc.stdout.decode('utf-8', errors='replace')
        # Also inspect uncompressed content/metadata streams, in addition to
        # Poppler's decoded page text and PDF metadata.
        for match in re.finditer(rb'\bstream\r?\n(.*?)\r?\nendstream\b', raw, re.S):
            try:
                block = zlib.decompress(match[1])
            except zlib.error:
                continue
            yield 'pdf-deflate-stream', block.decode('utf-8', errors='replace')


def scan_tree(root):
    root = Path(root)
    hits = []
    count = pdfs = 0
    for path in sorted(root.rglob('*')):
        if path.relative_to(root).parts[0] == '.git':
            continue
        assert not path.is_symlink(), 'privacy scan rejects symbolic links'
        relative = path.relative_to(root).as_posix()
        for rule in matches(relative):
            hits.append(dict(file=relative, view='filename', rule=rule))
        if not path.is_file():
            continue
        count += 1
        pdfs += path.suffix.lower() == '.pdf'
        for view, content in file_views(path):
            for rule in matches(content):
                hits.append(dict(file=relative, view=view, rule=rule))
    return dict(status='PASS' if not hits else 'FAIL', files=count, pdfs=pdfs,
                hits=hits, hit_count=len(hits),
                scope='All filenames/raw bytes, UTF-16 views, PDF text/metadata/deflate streams; known patterns, not a guarantee against every possible private identifier.')


def check_redactions(root):
    root = Path(root)
    from redaction_contract import ORIGINALS, PACKET_VERSION
    data = json.loads((root/'REDACTIONS.json').read_text())
    assert data['packet_version'] == PACKET_VERSION
    assert data['status'] == 'REDACTED_PUBLIC_DERIVATIVES'
    assert data['originals_included'] is False
    assert data['proof_sources_changed'] is False and data['paper_bytes_changed'] is False
    assert data['records'], 'missing redaction records'
    expected = {n: (h, size) for n, h, size in ORIGINALS}
    assert len(data['records']) == len(expected)
    seen = set()
    table = (root/'REDACTIONS.md').read_text()
    for record in data['records']:
        name = record['public_file']
        assert name not in seen and name in expected
        seen.add(name)
        assert (record['original_sha256'], record['original_bytes']) == expected[name], 'redaction original hash contract'
        relative = Path(name)
        assert not relative.is_absolute() and '..' not in relative.parts
        path = root/relative
        raw = path.read_bytes()
        assert (hashlib.sha256(raw).hexdigest(), len(raw)) == (record['public_sha256'], record['public_bytes']), 'redaction public hash mismatch'
        assert record['original_sha256'] != record['public_sha256']
        assert record['rule_ids'] and set(record['rule_ids']) <= set(data['rules'])
        assert record['public_sha256'] in table and record['original_sha256'] in table and name in table, 'redaction Markdown correspondence'
        assert not any(hashlib.sha256(p.read_bytes()).hexdigest() == record['original_sha256']
                       for p in root.rglob('*') if p.is_file() and p.relative_to(root).parts[0] != '.git'), 'unredacted original bundled'
    assert seen == set(expected)
    return dict(redacted_records=len(seen), originals_included=False)


def check_scanner_source(root):
    # Public structural policy: no identifier dictionaries or private fingerprints.
    source = (Path(root)/'publication_privacy.py').read_text()
    tree = ast.parse(source)
    forbidden_name = 'PRIVATE_IDENTIFIER_' + 'HASHES'
    for node in ast.walk(tree):
        assert not (isinstance(node, ast.Name) and node.id == forbidden_name), 'private identifier registry restored'
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            assert not re.fullmatch(r'[0-9a-fA-F]{64}', node.value), 'private fingerprint literal restored'
    return dict(public_scanner='generic-patterns-only', private_identifier_registry_bundled=False)


def check_privacy(root):
    # Run the scan first: restored original records must fail on private content,
    # rather than only on a derived fingerprint mismatch.
    structural = check_scanner_source(root)
    scan = scan_tree(root)
    assert not scan['hits'], 'publication privacy scan detected private content: ' + json.dumps(scan['hits'], sort_keys=True)
    receipt = check_redactions(root)
    return dict(privacy_scan=scan, redactions=receipt, scanner_policy=structural)


"""Context adapter for generic privacy scans: source operators and gzip data."""
import io
import tokenize
import gzip

_base_file_views = file_views


def _code_view(text):
    """Mask only Python operator/attribute tokens, never strings or comments."""
    lines = text.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))
    tokens = list(tokenize.generate_tokens(io.StringIO(text).readline))
    spans = []
    for i, token in enumerate(tokens):
        operator = token.type == tokenize.OP and token.string == '@'
        method = (token.type == tokenize.NAME and token.string == 'local'
                  and i > 0 and tokens[i-1].string == '.'
                  and i+1 < len(tokens) and tokens[i+1].string == '(')
        if operator or method:
            spans.append((offsets[token.start[0]-1]+token.start[1],
                          offsets[token.end[0]-1]+token.end[1]))
    for start, end in reversed(spans):
        text = text[:start] + ' '*(end-start) + text[end:]
    return text


def file_views(path):
    if path.suffix.lower() == '.gz':
        # Flags0 means no user-controlled filename/comment/extra header fields.
        raw = path.read_bytes()
        assert raw[:3] == bytes([31, 139, 8]) and raw[3] == 0
        with gzip.open(path, 'rb') as stream:
            decoded = stream.read(128*1024*1024+1)
        assert len(decoded) <= 128*1024*1024, 'privacy decode size bound'
        yield 'gzip-decoded-utf8', decoded.decode('utf-8')
        return
    for view, text in _base_file_views(path):
        if path.suffix.lower() == '.py' and view == 'raw-utf8':
            yield 'python-source-strings-comments-and-tokens', _code_view(text)
        else:
            yield view, text


if __name__ == '__main__':
    result = scan_tree(Path(sys.argv[1]))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    sys.exit(0 if result['status'] == 'PASS' else 1)
