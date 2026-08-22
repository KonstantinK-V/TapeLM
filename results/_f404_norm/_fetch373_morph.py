"""German / Finnish running text and real code identifiers for 373."""
from __future__ import annotations
import io
import json
import keyword
import re
import tarfile
import urllib.request
import zipfile
from pathlib import Path
DATA = Path('data')
UA = 'sote-letter-assembly/373-morph (research; local Windows)'
IDENT = re.compile('[A-Za-z_][A-Za-z0-9_]*')
KW = set(keyword.kwlist) | {'self', 'cls', 'None', 'True', 'False'}

def get(url: str, timeout: int=180) -> bytes:
    req = urllib.request.Request(url, headers={'User-Agent': UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()

def write_long_lines(texts, dest: Path, min_len: int=80, max_lines: int=12000) -> int:
    dest.parent.mkdir(parents=True, exist_ok=True)
    out = []
    for t in texts:
        t = ' '.join(t.split())
        if len(t) >= min_len:
            out.append(t)
        if len(out) >= max_lines:
            break
    dest.write_text('\n'.join(out) + '\n', encoding='utf-8')
    print(f'wrote {dest}  {len(out)} lines')
    return len(out)

def leipzig_sentences(url: str, dest: Path) -> int:
    raw = get(url)
    with tarfile.open(fileobj=io.BytesIO(raw), mode='r:gz') as tar:
        sent = next((m for m in tar.getmembers() if m.isfile() and m.name.endswith('sentences.txt')), None)
        if sent is None:
            raise RuntimeError('no sentences.txt in ' + url)
        f = tar.extractfile(sent)
        assert f is not None
        texts = []
        for line in f:
            t = line.decode('utf-8', 'ignore').strip()
            if '\t' in t:
                t = t.split('\t', 1)[-1]
            texts.append(t)
    return write_long_lines(texts, dest)

def wiki_extracts(lang: str, dest: Path, n_pages: int=400) -> int:
    lines = []
    url = f'https://{lang}.wikipedia.org/w/api.php?action=query&format=json&generator=random&grnnamespace=0&grnlimit=8&prop=extracts&explaintext=1&exsectionformat=plain'
    while len(lines) < n_pages:
        data = json.loads(get(url, timeout=60).decode('utf-8', 'ignore'))
        pages = (data.get('query') or {}).get('pages') or {}
        if not pages:
            break
        for p in pages.values():
            text = (p.get('extract') or '').replace('\n', ' ').strip()
            if len(text) >= 80:
                lines.append(text)
    return write_long_lines(lines, dest, max_lines=n_pages)

def opus_wiki(lang: str, dest: Path) -> int:
    import gzip
    url = f'https://object.pouta.csc.fi/OPUS-Wikipedia/v1.0/mono/{lang}.txt.gz'
    raw = gzip.decompress(get(url, timeout=300))
    texts = raw.decode('utf-8', 'ignore').splitlines()
    return write_long_lines(texts, dest)

def pypi_sdist_url(name: str) -> str:
    meta = json.loads(get(f'https://pypi.org/pypi/{name}/json', timeout=60).decode())
    for item in meta['urls']:
        if item.get('packagetype') == 'sdist' and item['url'].endswith(('.tar.gz', '.zip')):
            return item['url']
    raise RuntimeError('no sdist for ' + name)

def identifiers_from_bytes(raw: bytes) -> list[str]:
    try:
        text = raw.decode('utf-8')
    except UnicodeDecodeError:
        text = raw.decode('latin-1', 'ignore')
    out = []
    for tok in IDENT.findall(text):
        if tok in KW or len(tok) < 2:
            continue
        out.append(tok)
    return out

def pack_idents(idents: list[str], dest: Path, per_line: int=12) -> int:
    lines = []
    buf = []
    for tok in idents:
        buf.append(tok)
        if len(buf) >= per_line:
            line = ' '.join(buf)
            if len(line) >= 80:
                lines.append(line)
            buf = []
            if len(lines) >= 12000:
                break
    if buf:
        line = ' '.join(buf)
        if len(line) >= 80:
            lines.append(line)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'wrote {dest}  {len(lines)} identifier lines, {len(set(idents))} distinct seen')
    return len(lines)

def code_from_packages(dest: Path) -> int:
    packages = ['requests', 'flask', 'werkzeug', 'jinja2', 'click', 'httpx', 'pydantic', 'rich', 'typer', 'attrs', 'idna', 'urllib3']
    idents: list[str] = []
    skip_dir = {'data', 'out', 'results', '.git', 'legacy', 'artifact', 'huggingface'}
    for p in Path('.').rglob('*.py'):
        if any((part in skip_dir for part in p.parts)):
            continue
        try:
            idents.extend(identifiers_from_bytes(p.read_bytes()))
        except OSError:
            pass
    for name in packages:
        try:
            url = pypi_sdist_url(name)
            raw = get(url, timeout=180)
            print(f'pypi {name}  {len(raw)} bytes')
        except Exception as e:
            print(f'pypi {name} failed: {e}')
            continue
        bio = io.BytesIO(raw)
        if url.endswith('.zip'):
            zf = zipfile.ZipFile(bio)
            for info in zf.infolist():
                if info.filename.endswith('.py'):
                    idents.extend(identifiers_from_bytes(zf.read(info)))
        else:
            with tarfile.open(fileobj=bio, mode='r:gz') as tar:
                for m in tar.getmembers():
                    if m.isfile() and m.name.endswith('.py'):
                        f = tar.extractfile(m)
                        if f is not None:
                            idents.extend(identifiers_from_bytes(f.read()))
    return pack_idents(idents, dest)

def try_sources(dest: Path, label: str, jobs: list) -> None:
    for name, fn in jobs:
        try:
            n = fn()
            if n >= 800:
                print(f'{label}: {name} ok ({n})')
                return
            print(f'{label}: {name} too small ({n}), next')
        except Exception as e:
            print(f'{label}: {name} failed: {e}')
    raise SystemExit(f'{label}: no corpus')
if __name__ == '__main__':
    de = DATA / '_morph_de.txt'
    fi = DATA / '_morph_fi.txt'
    code = DATA / '_morph_code_idents.txt'
    try_sources(de, 'de', [('leipzig', lambda: leipzig_sentences('https://downloads.wortschatz-leipzig.de/corpora/deu_news_2024_100K.tar.gz', de)), ('opus', lambda: opus_wiki('de', de)), ('wiki-api', lambda: wiki_extracts('de', de))])
    try_sources(fi, 'fi', [('leipzig', lambda: leipzig_sentences('https://downloads.wortschatz-leipzig.de/corpora/fin_news_2024_100K.tar.gz', fi)), ('opus', lambda: opus_wiki('fi', fi)), ('wiki-api', lambda: wiki_extracts('fi', fi))])
    n = code_from_packages(code)
    if n < 800:
        raise SystemExit(f'code corpus too small: {n}')