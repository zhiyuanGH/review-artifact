"""Differential tool validation on a host binary or the actual Android device."""
import argparse, copy, json, shlex, subprocess, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from prepare import save, params, reference, digest

def equal(a, b):
    if isinstance(a, (int, float)) and (not isinstance(a, bool)) and isinstance(b, (int, float)) and (not isinstance(b, bool)):
        return abs(a - b) < 1e-08
    if type(a) != type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all((equal(a[k], b[k]) for k in a))
    if isinstance(a, list):
        return len(a) == len(b) and all((equal(x, y) for x, y in zip(a, b)))
    return a == b

class Target:

    def __init__(self, args):
        self.args = args
        self.root = args.output.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.adb = [str(args.adb), '-s', args.serial] if args.serial else None
        self.remote = args.remote
        self.binary = str(args.binary.resolve())
        self.traces = []
        if self.adb:
            self.shell('mkdir -p ' + shlex.quote(self.remote))
            self.push(args.binary, self.remote + '/wfgen-tools')
            self.shell('chmod 755 ' + shlex.quote(self.remote + '/wfgen-tools'))
            for domain in ['retail', 'airline']:
                self.push(args.bundle / 'initial' / f'{domain}.json', self.remote + f'/{domain}.initial.json')
            self.binary = self.remote + '/wfgen-tools'
            self.device = self.shell('getprop ro.product.model; getprop ro.soc.model; getprop ro.build.version.sdk').strip().splitlines()
        else:
            self.device = ['host']

    def shell(self, command):
        return subprocess.run(self.adb + ['shell', command], capture_output=True, text=True, check=True, timeout=60).stdout

    def push(self, path, remote):
        subprocess.run(self.adb + ['push', str(path), remote], capture_output=True, check=True, timeout=60)

    def reset(self, domain, name):
        if self.adb:
            path = self.remote + '/' + name + '.state.json'
            self.shell('cp ' + shlex.quote(self.remote + f'/{domain}.initial.json') + ' ' + shlex.quote(path))
        else:
            path = str(self.root / (name + '.state.json'))
            Path(path).write_bytes((self.args.bundle / 'initial' / f'{domain}.json').read_bytes())
        return path

    def read(self, path):
        return json.loads(self.shell('cat ' + shlex.quote(path)) if self.adb else Path(path).read_text())

    def call(self, tool, p, state, upstream=None):
        command = [self.binary, tool, '--state', state]
        if self.adb:
            command = self.adb + ['shell', shlex.join(command)]
        request = {'entrypoint': tool, 'parameters': p, 'upstream': upstream or {}}
        started = time.perf_counter()
        r = subprocess.run(command, input=json.dumps(request), capture_output=True, text=True, timeout=60)
        wall = (time.perf_counter() - started) * 1000
        response = json.loads(r.stdout)
        assert r.returncode == (0 if response['success'] else 1), (r.returncode, response)
        if self.adb:
            assert response['raw']['platform'] == 'android'
        record = {'command': command, 'request': request, 'response': response, 'host_wall_ms': wall, 'returncode': r.returncode}
        self.traces.append(record)
        return response

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bundle', type=Path, required=True)
    ap.add_argument('--binary', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--adb', type=Path)
    ap.add_argument('--serial')
    ap.add_argument('--remote', default='/data/local/tmp/wfgen_native_tools/v1')
    args = ap.parse_args()
    t = Target(args)
    doc = json.loads((args.bundle / 'workflows.json').read_text())
    refroot = args.bundle / 'reference_source'
    results = []
    seen = {}
    try:
        for w in doc['workflows']:
            idx = w['metadata']['selection_index']
            domain = 'airline' if '_airline_' in w['workflow_id'] else 'retail'
            state = t.reset(domain, f'validation_{idx}')
            reference_state = json.loads((args.bundle / 'initial' / f'{domain}.json').read_text())
            steps = []
            for n in w['nodes']:
                if n['kind'] not in ['app', 'dnn']:
                    continue
                tool = n['entrypoint']
                p = params(n)
                seen.setdefault(tool, (domain, p))
                expected = reference(refroot, domain, tool, reference_state['data'], p)
                actual = t.call(tool, p, state)
                assert actual['success'] == expected['success'], (idx, tool, actual, expected)
                if expected['success']:
                    assert equal(actual['raw']['result'], expected['result']), (idx, tool, 'output mismatch')
                else:
                    assert actual['error'] == expected['error'], (idx, tool, actual, expected)
                assert equal(t.read(state), reference_state), (idx, tool, 'state mismatch')
                steps.append({'tool': tool, 'success': actual['success'], 'output_matches': True, 'state_matches': True})
                if not actual['success']:
                    break
            save(args.output / f'{idx}.state.json', t.read(state))
            results.append({'source_index': idx, 'steps': steps, 'all_success': all((x['success'] for x in steps)), 'reference_match': True})
            print('workflow', idx, 'tools', len(steps), 'passed', results[-1]['all_success'], flush=True)
        negatives = []
        for tool, (domain, p) in sorted(seen.items()):
            p = copy.deepcopy(p)
            key = 'email' if tool == 'find_user_by_contact' else 'first_name' if tool == 'find_user_by_name' else 'user_id' if tool in ['get_user_details', 'modify_user_address'] else 'product_id' if tool == 'get_product_details' else 'reservation_id' if tool.startswith('update_reservation') else 'order_id'
            p[key] = '__missing__'
            state = t.reset(domain, 'negative_' + tool)
            before = t.read(state)
            actual = t.call(tool, p, state)
            assert not actual['success']
            assert equal(t.read(state), before)
            negatives.append(tool)
        for idx, tool in [(7, 'cancel_order'), (33, 'return_delivered_order_items')]:
            w = next((w for w in doc['workflows'] if w['metadata']['selection_index'] == idx))
            n = next((n for n in w['nodes'] if n['entrypoint'] == tool))
            state = t.reset('retail', 'repeat_' + str(idx))
            assert t.call(tool, params(n), state)['success']
            before = t.read(state)
            assert not t.call(tool, params(n), state)['success']
            assert equal(before, t.read(state))
        w = next((w for w in doc['workflows'] if w['metadata']['selection_index'] == 152))
        nodes = [n for n in w['nodes'] if n['entrypoint'] in ['cancel_order', 'return_delivered_order_items']]
        state = t.reset('retail', 'concurrent')
        refstate = t.read(state)
        for n in nodes:
            assert reference(refroot, 'retail', n['entrypoint'], refstate['data'], params(n))['success']
        with ThreadPoolExecutor(max_workers=2) as pool:
            concurrent = list(pool.map(lambda n: t.call(n['entrypoint'], params(n), state), nodes))
        assert all((r['success'] for r in concurrent)) and equal(t.read(state), refstate)
        policy_cases = [('modify_pending_order_payment', {'payment_method_id': '__missing__'}), ('exchange_delivered_order_items', {'new_item_ids': ['__missing__', '__missing__']}), ('update_reservation_flights', {'payment_id': '__missing__'}), ('update_reservation_baggages', {'payment_id': '__missing__'})]
        for tool, overrides in policy_cases:
            domain, p = seen[tool]
            p = {**p, **overrides}
            state = t.reset(domain, 'policy_' + tool)
            before = t.read(state)
            assert not t.call(tool, p, state)['success']
            assert equal(before, t.read(state))
        report = {'device': t.device, 'serial': args.serial, 'binary_sha256': digest(args.binary), 'selection_sha256': digest(args.bundle / 'workflows.json'), 'workflows': results, 'negative_tools': negatives, 'repeat_checks': 2, 'policy_rollback_checks': len(policy_cases), 'concurrent_updates_match': True, 'all_checks_passed': True, 'calls': len(t.traces)}
        save(args.output / 'report.json', report)
        print(json.dumps({k: v for k, v in report.items() if k != 'workflows'}, ensure_ascii=False), flush=True)
    finally:
        save(args.output / 'calls.json', t.traces)
if __name__ == '__main__':
    main()
