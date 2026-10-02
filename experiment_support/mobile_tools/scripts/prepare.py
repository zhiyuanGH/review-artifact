"""Freeze the length-spaced sample, original databases, and independent reference outputs."""
import argparse, copy, hashlib, importlib.util, json, shutil, sys, types
from pathlib import Path

def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def save(p, d):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(d, ensure_ascii=False, indent=2) + '\n')

def params(n):
    out = {}
    for p in n.get('parameters', []):
        v = p['value']
        if isinstance(v, str) and v.strip().startswith(('[', '{')):
            try:
                v = json.loads(v)
            except ValueError:
                pass
        if p['name'] in ('total_baggages', 'nonfree_baggages'):
            v = int(v)
        out[p['name']] = v
    return out
MAPPING = {'find_user_by_contact': ('find_user_id_by_email', 'FindUserIdByEmail'), 'find_user_by_name': ('find_user_id_by_name_zip', 'FindUserIdByNameZip'), 'get_user_details': ('get_user_details', 'GetUserDetails'), 'get_order_details': ('get_order_details', 'GetOrderDetails'), 'get_product_details': ('get_product_details', 'GetProductDetails'), 'cancel_order': ('cancel_pending_order', 'CancelPendingOrder')}

def reference(source, domain, tool, data, p):
    for name in ['tau_bench', 'tau_bench.envs']:
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)
    name = 'tau_bench.envs.tool'
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, source / 'tool.py')
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    filename, cls = MAPPING.get(tool, (tool, ''.join((x.title() for x in tool.split('_')))))
    p = copy.deepcopy(p)
    if domain == 'airline' and tool in ['get_order_details', 'cancel_order']:
        filename, cls = ('get_reservation_details', 'GetReservationDetails') if tool == 'get_order_details' else ('cancel_reservation', 'CancelReservation')
        p = {'reservation_id': p['order_id']}
    spec = importlib.util.spec_from_file_location(f'ref_{domain}_{filename}', source / domain / 'tools' / f'{filename}.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    raw = getattr(mod, cls).invoke(data, **p)
    if raw.startswith('Error:'):
        return {'success': False, 'error': raw}
    try:
        result = json.loads(raw)
    except ValueError:
        result = raw
    return {'success': True, 'result': result}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workflow', type=Path, required=True)
    ap.add_argument('--tau-root', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    source = args.tau_root / 'tau_bench/envs'
    doc = json.loads(args.workflow.read_text())
    rows = sorted(doc['workflows'], key=lambda w: (sum((n['kind'] in ['app', 'dnn'] for n in w['nodes'])), w['metadata']['selection_index'], w['workflow_id']))
    empty = [w for w in rows if not w['nodes']]
    population = [w for w in rows if any((n['kind'] in ['app', 'dnn'] for n in w['nodes']))]
    ranks = [(2 * i * (len(population) - 1) + 9) // 18 for i in range(10)]
    chosen = [population[k] for k in ranks]
    selected = copy.deepcopy(doc)
    selected['workflows'] = chosen
    selected['benchmark']['task_count'] = 10
    selected['selection'] = {'population': 'nonempty compact280 workflows', 'population_count': len(population), 'excluded_empty': len(empty), 'sort': '(app+dnn node count,source selection_index,workflow_id)', 'ranks_zero_based': ranks, 'source_indices': [w['metadata']['selection_index'] for w in chosen], 'source_sha256': digest(args.workflow), 'rule': 'floor(i*(N-1)/9+0.5), i=0..9'}
    save(out / 'workflows.json', selected)
    save(out / 'excluded_empty.json', {'workflow_ids': [w['workflow_id'] for w in empty]})
    database_hashes = {}
    initial = {}
    for domain, tables in [('retail', ['users', 'orders', 'products']), ('airline', ['users', 'reservations', 'flights'])]:
        initial[domain] = {'domain': domain, 'data': {}}
        for table in tables:
            p = source / domain / 'data' / f'{table}.json'
            initial[domain]['data'][table] = json.loads(p.read_text())
            database_hashes[f'{domain}/{table}'] = {'source': str(p), 'sha256': digest(p)}
        save(out / 'initial' / f'{domain}.json', initial[domain])
    refroot = out / 'reference_source'
    refroot.mkdir(exist_ok=True)
    shutil.copy2(source / 'tool.py', refroot / 'tool.py')
    shutil.copy2(args.tau_root / 'LICENSE', refroot / 'LICENSE')
    for domain in ['retail', 'airline']:
        shutil.copytree(source / domain / 'tools', refroot / domain / 'tools', ignore=shutil.ignore_patterns('__pycache__'), dirs_exist_ok=True)
    reports = []
    for w in chosen:
        domain = 'airline' if '_airline_' in w['workflow_id'] else 'retail'
        state = copy.deepcopy(initial[domain])
        steps = []
        for n in w['nodes']:
            if n['kind'] not in ['app', 'dnn']:
                continue
            response = reference(refroot, domain, n['entrypoint'], state['data'], params(n))
            steps.append({'node_id': n['node_id'], 'entrypoint': n['entrypoint'], 'parameters': params(n), 'reference': response})
            if not response['success']:
                break
        index = w['metadata']['selection_index']
        save(out / 'expected' / f'{index}.state.json', state)
        save(out / 'expected' / f'{index}.steps.json', steps)
        reports.append({'source_index': index, 'domain': domain, 'tool_steps': len(steps), 'success': all((s['reference']['success'] for s in steps)), 'errors': [s['reference'] for s in steps if not s['reference']['success']]})
    source_hashes = {str(p.relative_to(refroot)): digest(p) for p in refroot.rglob('*.py')}
    save(out / 'provenance.json', {'selection': selected['selection'], 'database_files': database_hashes, 'reference_source_hashes': source_hashes, 'workflow_sha256': digest(out / 'workflows.json'), 'initial_sha256': {d: digest(out / 'initial' / f'{d}.json') for d in initial}, 'reference_results': reports, 'notes': ['Original tau-bench database snapshot, not state inferred from reference actions.', 'Fixed-action replay does not validate dialogue/user confirmation or prompt-only planning.', 'Tool aliases adapt WFGen names to pinned tau-bench invoke functions.']})
    print(json.dumps({'selection': selected['selection'], 'reference_results': reports}, ensure_ascii=False))
if __name__ == '__main__':
    main()
