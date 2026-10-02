"""Run the frozen ten workflows with phone-native tools and real phone LLM checkers."""
import argparse, copy, json, shlex, shutil, subprocess, sys, time
from pathlib import Path
from prepare import save, digest
from validate import equal
from scheduling import derive_plan

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bundle', type=Path, required=True)
    ap.add_argument('--framework-root', '--wfgen', dest='wfgen', type=Path, required=True)
    ap.add_argument('--adb', type=Path, required=True)
    ap.add_argument('--serial', required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--mode', choices=['serial', 'parallel', 'both'], default='both')
    ap.add_argument('--model', default='/data/local/tmp/gguf/qwen1_5-1_8b-chat-q4_0.gguf')
    ap.add_argument('--remote', default='/data/local/tmp/wfgen_native_tools/v1')
    ap.add_argument('--source-index', type=int, action='append', help='Run only these frozen source indices; repeatable.')
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    runtime_root = args.output / 'runtime_source'
    package = runtime_root / 'MOBench'
    hashes = {}
    for rel in ['workflow_execution/runtime.py', 'tools/llm_tau.py', 'tools/abi.py']:
        dest = package / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.wfgen / rel, dest)
        hashes[rel] = digest(dest)
    for folder in [package, package / 'tools', package / 'workflow_execution']:
        (folder / '__init__.py').write_text('')
    sys.path.insert(0, str(runtime_root.resolve()))
    from MOBench.workflow_execution.runtime import MobileWorkflowRuntime
    adb = [str(args.adb.resolve()), '-s', args.serial]

    def shell(s):
        return subprocess.run(adb + ['shell', s], capture_output=True, text=True, timeout=60, check=True).stdout
    binary_hash = shell('sha256sum ' + shlex.quote(args.remote + '/wfgen-tools')).split()[0]
    doc = json.loads((args.bundle / 'workflows.json').read_text())
    results = []
    selected = [w for w in doc['workflows'] if args.source_index is None or w['metadata']['selection_index'] in args.source_index]
    if not selected or (args.source_index is not None and set(args.source_index) != {w['metadata']['selection_index'] for w in selected}):
        raise ValueError('Requested source indices must belong to the frozen sample')
    for mode in ['serial', 'parallel'] if args.mode == 'both' else [args.mode]:
        for original in selected:
            w, added_dependencies = derive_plan(original, MobileWorkflowRuntime._topological_waves)
            idx = w['metadata']['selection_index']
            domain = 'airline' if '_airline_' in w['workflow_id'] else 'retail'
            name = f'{mode}_{idx}'
            state = args.remote + '/' + name + '.state.json'
            shell('cp ' + shlex.quote(args.remote + f'/{domain}.initial.json') + ' ' + shlex.quote(state))
            initial = json.loads(shell('cat ' + shlex.quote(state)))
            assert equal(initial, json.loads((args.bundle / 'initial' / f'{domain}.json').read_text()))
            if mode == 'serial':
                ordering = [node for wave in w['execution_hints']['parallel_waves'] for node in wave['node_ids']]
                w['execution_hints']['parallel_waves'] = [{'wave_index': i, 'node_ids': [node], 'reason': 'Serial ablation of the same DAG.'} for i, node in enumerate(ordering)]
            llm = {'llama_cli_target': 'phone', 'adb': str(args.adb.resolve()), 'device_serial': args.serial, 'remote_dir': '/data/local/tmp/llama.cpp', 'model_path': args.model, 'llama_device': 'none', 'ngl': 0, 'hexagon_env': False, 'max_tokens': 64, 'temperature': 0.0, 'llama_timeout': 180, 'threads': 6, 'json_schema': '{}'}
            cfg = {'max_parallel_nodes': 1 if mode == 'serial' else 2, 'llm': {'runtime': 'llama_cli', 'llama_cli': llm}, 'qnn_tools': {}}
            for n in w['nodes']:
                if n['kind'] not in ['app', 'dnn']:
                    continue
                tool = n['entrypoint']
                cfg['qnn_tools'][tool] = {'mode': 'json_cli', 'command': adb + ['shell', shlex.join([args.remote + '/wfgen-tools', tool, '--state', state])], 'timeout': 60, 'deployment': {'selected_runtime': 'phone_native_json_cli', 'device': 'phone', 'device_class': 'cpu'}}
            save(args.output / f'{name}.runtime.json', cfg)
            save(args.output / f'{name}.workflow.json', w)
            print('START', name, flush=True)
            r = MobileWorkflowRuntime(cfg).execute_workflow(w)
            actual = json.loads(shell('cat ' + shlex.quote(state)))
            expected = json.loads((args.bundle / 'expected' / f'{idx}.state.json').read_text())
            state_match = equal(actual, expected)
            reference_steps = {s['node_id']: s['reference'] for s in json.loads((args.bundle / 'expected' / f'{idx}.steps.json').read_text())}
            tools = [s for s in r['steps'] if s['kind'] in ['app', 'dnn']]
            llms = [s for s in r['steps'] if s['kind'] == 'llm']
            phone_tools = all((s['raw'].get('response', {}).get('raw', {}).get('platform') == 'android' for s in tools))
            tool_outputs_match = all((equal(s['raw'].get('response', {}).get('raw', {}).get('result'), reference_steps[s['node_id']].get('result')) for s in tools))
            summary = {'mode': mode, 'source_index': idx, 'status': r['status'], 'state_matches_reference': state_match, 'tool_outputs_match_reference': tool_outputs_match, 'phone_native_tools': phone_tools, 'tool_count': len(tools), 'llm_count': len(llms), 'all_llm_on_phone': all((s['raw'].get('target') == 'phone' for s in llms)), 'host_workflow_wall_ms': r['total_wall_ms']}
            r['validation'] = summary
            r['provenance'] = {'binary_sha256': binary_hash, 'workflow_sha256': digest(args.bundle / 'workflows.json'), 'execution_plan_sha256': digest(args.output / f'{name}.workflow.json'), 'added_state_dependencies': added_dependencies, 'initial_state_sha256': digest(args.bundle / 'initial' / f'{domain}.json'), 'device_serial': args.serial, 'runtime_hashes': hashes, 'limitation': 'Host orchestrates DAG; checker process success is not an independent semantic verdict.'}
            save(args.output / f'{name}.result.json', r)
            save(args.output / f'{name}.state.json', actual)
            results.append(summary)
            save(args.output / 'summary.json', {'results': results, 'runtime_hashes': hashes, 'complete': False})
            print('DONE', json.dumps(summary), flush=True)
            if not (r['status'] == 'completed' and state_match and phone_tools and tool_outputs_match and summary['all_llm_on_phone']):
                raise RuntimeError('Integration validation failed: ' + name)
    save(args.output / 'summary.json', {'results': results, 'runtime_hashes': hashes, 'binary_sha256': binary_hash, 'device_serial': args.serial, 'complete': True, 'all_passed': True, 'scope': 'Implementation validation only; no repeated performance study or checker semantic-accuracy claim.'})
if __name__ == '__main__':
    main()
