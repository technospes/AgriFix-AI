import json
from pathlib import Path
from collections import Counter

base = Path('mahindra_pdfs')

# Main DB
db_file = base / 'Master_Tractors_DB.json'
if db_file.exists():
    with open(db_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f'{"="*60}')
    print(f'📊 Master_Tractors_DB.json: {len(data)} procedures')
    
    families = Counter(p.get('machine_family', 'Unknown') for p in data)
    for fam, count in families.most_common():
        print(f'   {fam}: {count}')
    
    types = Counter(p.get('knowledge_type', 'Unknown') for p in data)
    print(f'   direct_repair: {types.get("direct_repair", 0)}')
    print(f'   causal_inferred: {types.get("causal_inferred", 0)}')

# Other DBs
files = {
    'fault_library': 'faults',
    'diagnostic_trees': 'trees',
    'repair_procedures': 'repair procs',
    'spec_database': 'specs',
    'tables': 'tables',
    'images': 'images',
    'component_graph': 'component nodes',
}

for suffix, label in files.items():
    fpath = base / f'Master_Tractors_DB_{suffix}.json'
    if fpath.exists():
        with open(fpath, 'r', encoding='utf-8') as f:
            content = json.load(f)
        count = len(content) if isinstance(content, list) else len(content) if isinstance(content, dict) else 0
        print(f'📄 Master_Tractors_DB_{suffix}.json: {count} {label}')

# Failed chunks
failed = Path('failed_chunks/failed_extractions.jsonl')
if failed.exists():
    with open(failed, 'r', encoding='utf-8') as f:
        lines = sum(1 for _ in f)
    print(f'\n⚠️  Failed chunks: {lines} (saved for re-processing)')

print(f'{"="*60}')