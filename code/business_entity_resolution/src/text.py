"""Conservative views: preserve Indic marks and never overwrite original text."""
import re
import unicodedata as ud

from anyascii import anyascii

SUFFIX = {'pvt': 'private', 'ltd': 'limited', 'corp': 'corporation', 'inc': 'incorporated'}
LEGAL = set(SUFFIX.values()) | {'llc', 'llp', 'co', 'company', 'plc', 'sarl', 'sas', 'sa'}
ADDRESS = {'rd': 'road', 'st': 'street', 'ave': 'avenue', 'blvd': 'boulevard', 'apt': 'apartment'}


def normalize(value):
    value = ud.normalize('NFKC', value).casefold()
    # Unicode combining marks are meaningful in native scripts.
    return ' '.join(''.join(c if ud.category(c)[0] in 'LNM' else ' ' for c in value).split())


def latin_fold(value):
    out = []
    latin = False
    for char in ud.normalize('NFD', value):
        if not ud.category(char).startswith('M'):
            latin = 'LATIN' in ud.name(char, '')
        if not (latin and ud.category(char).startswith('M')):
            out.append(char)
    return ud.normalize('NFC', ''.join(out))


def grams(value):
    value = value.replace(' ', '_')
    return sorted({value[i:i+3] for i in range(max(0, len(value)-2))})


def numbers(value):
    # Preserve compound numbers alongside individual numeric/alphanumeric pieces.
    return sorted(set(re.findall(r'\b\w*\d\w*(?:[-/]\w*\d\w*)*\b', value.casefold())))


def script(value):
    scripts = {ud.name(c, '').split(' ')[0] for c in value if c.isalpha()}
    return sorted(scripts)


def views(name, address):
    n, a = normalize(name), normalize(address)
    if a == 'null':
        a = ''
    canon = ' '.join(SUFFIX.get(t, t) for t in normalize(name.replace('&', ' and ')).split())
    core = ' '.join(t for t in canon.split() if t not in LEGAL)
    ac = ' '.join(ADDRESS.get(t, t) for t in a.split())
    return dict(n=n, a=a, nf=latin_fold(n), af=latin_fold(a),
                nt=normalize(anyascii(n)), at=normalize(anyascii(ac)), core=core,
                ac=ac, nums=numbers(address) if a else [], name_nums=numbers(name),
                scripts=script(name))


def fields(v):
    return {'name': sorted(set((v['n']+' '+v['nt']+' '+v['core']).split())),
            'gram': sorted(set(grams(v['n']) + grams(v['nt']))),
            'address': sorted(set((v['a']+' '+v['at']).split())),
            'number': v['nums']}
