"""Run selected native workflows on a phone; preserve failures and continue.

ODABench candidates are wave p=2 and p=3. The analysis selects one complete
configuration per device/model, never a best configuration per workflow.
"""
import argparse
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[3]
MODELS = {
    'qwen3_0_6b_q4_k_m': 'Qwen3-0.6B-Q4_K_M.gguf',
    'llama_3_2_1b_instruct_q4_0': 'Llama-3.2-1B-Instruct-Q4_0.gguf',
    'qwen1_5_1_8b_chat_q4_0': 'qwen1_5-1_8b-chat-q4_0.gguf',
}
CONFIGS = [('naive', 'cli-balanced-round-robin', 1),
           ('wave2', 'cli-balanced-wave-aware', 2),
           ('wave3', 'cli-balanced-wave-aware', 3),
           ('max_rr', 'cli-balanced-round-robin', 4),
           ('max_wave', 'cli-balanced-wave-aware', 4)]
INDICES = [120, 7, 33, 100, 186, 213, 239, 48, 152, 179]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(p, value):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(p)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', required=True, type=Path)
    ap.add_argument('--serial', required=True)
    ap.add_argument('--bundle', type=Path, default=ROOT / 'local_data/phone_tools_v1')
    ap.add_argument('--runtime-source', type=Path, default=ROOT)
    ap.add_argument('--binary', type=Path, required=True)
    ap.add_argument('--adb', default='adb')
    ap.add_argument('--device-profile', choices=['elite','gen3'], default='elite', help='Preserve the measured profile-specific configuration order.')
    ap.add_argument('--resource-tools-dir', type=Path, default=ROOT/'local_data/perfetto_tools')
    ap.add_argument('--remote', default='/data/local/tmp/wfgen_native_tools/selected10_four_20260928')
    args = ap.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault('NO_PROXY', 'localhost,127.0.0.1,::1')
    os.environ.setdefault('no_proxy', os.environ['NO_PROXY'])
    package = out / 'runtime/MOBench'
    shutil.copytree(args.runtime_source, package, ignore=shutil.ignore_patterns('__pycache__', '.git', 'local_data'))
    for name in ('scheduling.py', 'validate.py', 'prepare.py', 'run_comparison.py'):
        shutil.copy2(Path(__file__).parent / name, out / name)
    sys.path.insert(0, str(out))
    sys.path.insert(0, str(package.parent))
    from scheduling import derive_plan
    from validate import equal
    from MOBench.workflow_execution.runtime import MobileWorkflowRuntime
    from MOBench.deployment_planning.core import llm_device_assignments
    from MOBench.tools.llm_tau import LlamaCliClient
    from MOBench.resource_monitoring import PerfettoResourceMonitor, ResourceMonitorConfig
    import MOBench.resource_monitoring.perfetto_monitor as monitoring

    # Unique tracing and prompt paths, so cleanup cannot touch unrelated runs.
    token = f'selected10_{os.getpid()}_{time.time_ns()}'
    monitoring.REMOTE_TRACE_FILENAME = token + '.perfetto-trace'
    monitoring.REMOTE_CONFIG_FILENAME = token + '.pbtxt'
    remote = args.remote + '/' + token
    adb = [args.adb, '-s', args.serial]

    def shell(command, check=True):
        return subprocess.run(adb + ['shell', command], capture_output=True, text=True, timeout=60, check=check)

    assert shell('getprop ro.soc.model').stdout.strip() in ('SM8650', 'SM8750')
    if shell('pidof llama-cli; pidof perfetto; true').stdout.strip():
        raise RuntimeError('Phone has an active LLM or trace process; avoid interference')
    shell('mkdir -p ' + shlex.quote(remote + '/prompts'))
    subprocess.run(adb + ['push', str(args.binary), remote + '/wfgen-tools'], check=True, capture_output=True, timeout=60)
    shell('chmod 755 ' + shlex.quote(remote + '/wfgen-tools'))
    assert shell('sha256sum ' + shlex.quote(remote + '/wfgen-tools')).stdout.split()[0] == sha(args.binary)
    bundle = out / 'bundle'
    bundle.mkdir()
    for folder in ('initial', 'expected'):
        shutil.copytree(args.bundle / folder, bundle / folder)
    shutil.copy2(args.bundle / 'workflows.json', bundle / 'workflows.json')
    shutil.copy2(args.bundle / 'provenance.json', bundle / 'provenance.json')
    for domain in ('retail', 'airline'):
        subprocess.run(adb + ['push', str(bundle / 'initial' / f'{domain}.json'), remote + f'/{domain}.initial.json'], capture_output=True, check=True, timeout=60)
        assert shell('sha256sum ' + shlex.quote(remote + f'/{domain}.initial.json')).stdout.split()[0] == sha(bundle / 'initial' / f'{domain}.json')
    workflows = json.loads((bundle / 'workflows.json').read_text())['workflows']
    assert [w['metadata']['selection_index'] for w in workflows] == INDICES
    plans = [derive_plan(w, MobileWorkflowRuntime._topological_waves) for w in workflows]
    assert max(len(wave['node_ids']) for w, _ in plans for wave in w['execution_hints']['parallel_waves']) == 4
    source_hashes = {str(p.relative_to(package)): sha(p) for p in package.rglob('*.py')}
    provenance = {'device': args.serial, 'source_indices': INDICES, 'workflow_sha256': sha(bundle / 'workflows.json'),
                  'binary_sha256': sha(args.binary), 'runtime_hashes': source_hashes, 'configs': CONFIGS,
                  'models': MODELS, 'remote': remote, 'status': 'running', 'expected_runs': 150,
                  'scope': 'Native phone business tools and phone LLMs; host schedules dependency waves.',
                  'success_definition': 'All nodes complete, actual phone LLM generation, native tool outputs and full final state match fixed reference.',
                  'selection': 'Wave p=2/p=3 only, per-device/model suite: most successes, then lowest total attempted wall time.',
                  'parameters': {'max_tokens': 64, 'temperature': 0, 'threads': 6, 'llm_timeout_s': 240, 'warmup_runs_per_backend_per_model': 1,
                                 'monitor': 'perfetto', 'sample_interval_s': 1, 'buffer_mb': 128}}
    save(out / 'provenance.json', provenance)
    save(out / 'device_initial.json', {'output': shell('getprop ro.build.fingerprint; dumpsys battery; df -h /data/local/tmp').stdout})
    all_rows = []

    def persist():
        save(out / 'progress.json', {'complete': False, 'expected': 150, 'attempted': len(all_rows), 'succeeded': sum(r['success'] for r in all_rows), 'rows': all_rows})

    def cleanup_owned_llms():
        cleaned = []
        for pid in shell('pidof llama-cli', check=False).stdout.split():
            if not pid.isdigit():
                continue
            cmdline = shell(f'cat /proc/{pid}/cmdline', check=False).stdout
            if remote + '/prompts/' in cmdline:
                shell('kill -TERM ' + pid, check=False)
                cleaned.append(pid)
        return cleaned

    for model_index, (model, filename) in enumerate(MODELS.items()):
        model_path = '/data/local/tmp/gguf/' + filename
        base_llm = {'llama_cli_target': 'phone', 'adb': args.adb, 'device_serial': args.serial,
                    'remote_dir': '/data/local/tmp/llama.cpp', 'remote_prompt_dir': remote + '/prompts',
                    'model_path': model_path, 'max_tokens': 64, 'temperature': 0., 'llama_timeout': 240,
                    'threads': 6, 'json_schema': '{}', 'warmup_runs': 1}
        warmups = []
        for device in ('none', 'GPUOpenCL', 'HTP0'):
            cfg = dict(base_llm, llama_device=device, ngl=0 if device == 'none' else 99, hexagon_env=device == 'HTP0')
            warmups.append({'backend': device, 'result': LlamaCliClient.from_config(cfg).warmup()})
            cleanup_owned_llms()
        save(out / model / 'warmups.json', warmups)
        # Rotate configuration order; opposite phone reverses it. Record actual order.
        order = CONFIGS[model_index:] + CONFIGS[:model_index]
        if args.device_profile == 'gen3':
            order = list(reversed(order))
        save(out / model / 'configuration_order.json', order)
        for config_id, strategy, parallel in order:
            case = out / model / config_id
            case.mkdir(parents=True)
            monitor = PerfettoResourceMonitor(ResourceMonitorConfig(adb_path=Path(args.adb), device_serial=args.serial,
                         output_dir=case / 'resources', tools_dir=args.resource_tools_dir, buffer_mb=128))
            monitor_records = []
            try:
                save(case / 'monitor_start.json', monitor.start())
                for plan, added in plans:
                    w = copy.deepcopy(plan)
                    idx = w['metadata']['selection_index']
                    prefix = case / str(idx)
                    domain = 'airline' if '_airline_' in w['workflow_id'] else 'retail'
                    state = remote + '/active.state.json'
                    row = {'device': args.serial, 'model_id': model, 'config': config_id, 'strategy': strategy,
                           'max_parallel_nodes': parallel, 'source_index': idx, 'workflow_id': w['workflow_id'],
                           'success': False, 'status': 'failed', 'failure_reason': '', 'wall_ms': None,
                           'state_matches_reference': False, 'tool_outputs_match_reference': False,
                           'result_json': str(prefix.with_suffix('.result.json'))}
                    started = time.time()
                    result = None
                    print('START', model, config_id, idx, flush=True)
                    try:
                        shell('cp ' + shlex.quote(remote + f'/{domain}.initial.json') + ' ' + shlex.quote(state))
                        assert shell('sha256sum ' + shlex.quote(state)).stdout.split()[0] == sha(bundle / 'initial' / f'{domain}.json')
                        placements = llm_device_assignments(w, strategy, runtime_defaults={'llama_model_path': model_path, 'adb_serial': args.serial, 'max_parallel_nodes': parallel})
                        overrides = {node: dict(base_llm, llama_device=device, ngl=0 if device == 'none' else 99, hexagon_env=device == 'HTP0') for node, device in placements.items()}
                        cfg = {'max_parallel_nodes': parallel, 'llm': {'runtime': 'llama_cli', 'llama_cli': dict(base_llm, llama_device='none', ngl=0), 'llama_cli_overrides': overrides}, 'qnn_tools': {}}
                        for node in w['nodes']:
                            if node['kind'] in ('app', 'dnn'):
                                tool = node['entrypoint']
                                cfg['qnn_tools'][tool] = {'mode': 'json_cli', 'command': adb + ['shell', shlex.join([remote + '/wfgen-tools', tool, '--state', state])], 'timeout': 60}
                        save(prefix.with_suffix('.runtime.json'), cfg)
                        save(prefix.with_suffix('.workflow.json'), {'workflow': w, 'added_dependencies': added})
                        started = time.time()
                        result = MobileWorkflowRuntime(cfg).execute_workflow(w)
                        row['wall_ms'] = result['total_wall_ms']
                        finished = time.time()
                        actual = json.loads(shell('cat ' + shlex.quote(state)).stdout)
                        save(prefix.with_suffix('.state.json'), actual)
                        expected = json.loads((bundle / 'expected' / f'{idx}.state.json').read_text())
                        refs = {s['node_id']: s['reference'] for s in json.loads((bundle / 'expected' / f'{idx}.steps.json').read_text())}
                        tool_steps = [s for s in result['steps'] if s['kind'] in ('app', 'dnn')]
                        llm_steps = [s for s in result['steps'] if s['kind'] == 'llm']
                        row['state_matches_reference'] = equal(actual, expected)
                        row['tool_outputs_match_reference'] = len(tool_steps) == len(refs) and all(equal(s['raw'].get('response', {}).get('raw', {}).get('result'), refs[s['node_id']].get('result')) for s in tool_steps)
                        phone_tools = all(s['raw'].get('response', {}).get('raw', {}).get('platform') == 'android' for s in tool_steps)
                        generated = len(llm_steps) == len(placements) and all(s['success'] and s['raw'].get('target') == 'phone' for s in llm_steps)
                        row['success'] = result['status'] == 'completed' and row['state_matches_reference'] and row['tool_outputs_match_reference'] and phone_tools and generated
                        row['status'] = 'completed' if row['success'] else 'failed'
                        errors = [f"{s['node_id']}: {s['raw'].get('error') or s['raw'].get('stderr') or s['raw'].get('response')}" for s in result['steps'] if not s['success']]
                        if not row['success'] and not errors:
                            errors = ['Reference state/output or phone execution validation failed']
                        row['failure_reason'] = '\n'.join(errors)
                    except Exception:
                        finished = time.time()
                        row['failure_reason'] = traceback.format_exc()
                        row['wall_ms'] = row['wall_ms'] if row['wall_ms'] is not None else (finished - started) * 1000
                    finally:
                        try:
                            row['cleaned_leftover_llm_pids'] = cleanup_owned_llms()
                            shell('rm -f ' + shlex.quote(state) + ' ' + shlex.quote(state + '.lock'), check=False)
                        except Exception:
                            row['cleanup_error'] = traceback.format_exc()
                            row['success'] = False
                            row['status'] = 'failed'
                    save(prefix.with_suffix('.result.json'), {'execution': result, 'summary': row})
                    all_rows.append(row)
                    monitor_records.append({'workflow_id': w['workflow_id'], 'workflow_index': idx, 'status': row['status'], 'started_epoch_s': started, 'finished_epoch_s': finished, 'result_json': row['result_json']})
                    persist()
                    print('DONE', len(all_rows), '/150', config_id, idx, row['status'], round(row['wall_ms']/1000, 2), flush=True)
            finally:
                try:
                    save(case / 'monitor_finish.json', monitor.stop(monitor_records))
                finally:
                    # A second stop signal is harmless if tracing already ended.
                    if monitor._perfetto_pid:
                        shell('kill -TERM ' + monitor._perfetto_pid, check=False)
                    shell('rm -f ' + shlex.quote(monitor.remote_trace_path) + ' ' + shlex.quote(monitor.remote_config_path), check=False)
    save(out / 'summary.json', {'complete': True, 'expected': 150, 'rows': all_rows})
    provenance['status'] = 'completed'
    provenance['successes'] = sum(r['success'] for r in all_rows)
    save(out / 'provenance.json', provenance)
    save(out / 'device_final.json', {'output': shell('dumpsys battery; df -h /data/local/tmp; pidof llama-cli; pidof perfetto; true').stdout})
    print('FINISHED', args.serial, provenance['successes'], '/150', flush=True)


if __name__ == '__main__':
    main()
