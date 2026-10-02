"""Prepare all missing selected-ten paper experiments; only --execute or
--execute-prepared uses phones. Existing audited measurements are reused,
including failures. No retries; resource repeats form a separate fixed cohort.
"""
import argparse
from collections import Counter
import copy
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time
import traceback

from run_comparison import INDICES, save, sha
from scheduling import derive_plan
from validate import equal
from selected10_matrix import cases, load_catalog, comparable_config, SUITES, QWEN

ROOT=Path(__file__).resolve().parents[3]
DEFAULT_RUNTIME=ROOT
DEVICE_A=os.environ.get('MOBENCH_DEVICE_A_SERIAL','DEVICE_A_SERIAL')
DEVICE_B=os.environ.get('MOBENCH_DEVICE_B_SERIAL','DEVICE_B_SERIAL')


def load_existing(root, bundle, binary, runtime_source):
    """Audit the fixed previous cohort; failures are reusable measurements too."""
    rows = {}
    for device in (DEVICE_A, DEVICE_B):
        folder = root / device
        summary = json.loads((folder/'summary.json').read_text())
        provenance = json.loads((folder/'provenance.json').read_text())
        if not summary['complete'] or len(summary['rows']) != 150:
            raise ValueError('Existing cohort is incomplete: '+device)
        if provenance['binary_sha256'] != sha(binary) or provenance['workflow_sha256'] != sha(bundle/'workflows.json'):
            raise ValueError('Existing cohort uses different tools/workflows')
        for relative, digest in provenance['runtime_hashes'].items():
            if sha(runtime_source/relative) != digest:
                raise ValueError('Runtime changed; cannot mix old/new measurements: '+relative)
        for path in list((bundle/'initial').glob('*.json'))+list((bundle/'expected').glob('*.json')):
            if sha(path) != sha(folder/'bundle'/path.relative_to(bundle)):
                raise ValueError('Reference state/output changed: '+str(path))
        for row in summary['rows']:
            key = (device, row['model_id'], row['config'], row['source_index'])
            if key in rows: raise ValueError('Duplicate historical record')
            source = Path(row['result_json'])
            if json.loads(source.read_text())['summary'] != row:
                raise ValueError('Summary differs from original record')
            rows[key] = row
    return rows


def runtime_config(case,workflow,serial,adb,remote,assign):
    common={'adb':adb,'device_serial':serial,'max_tokens':64,'timeout':240}
    ids=[n['node_id'] for n in workflow['nodes'] if n['kind']=='llm']
    if case['runtime']=='llama_cli':
        base=dict(common,llama_cli_target='phone',remote_dir='/data/local/tmp/llama.cpp',
                  remote_prompt_dir=remote+'/prompts',model_path=case['model_path'],
                  temperature=0.,llama_timeout=240,threads=6,json_schema='{}',warmup_runs=1)
        fixed={'npu':'HTP0','cpu':'none','gpu':'GPUOpenCL'}
        placement={n:fixed[case['placement']] for n in ids} if case['placement'] in fixed else assign(
            workflow,'cli-balanced-round-robin' if case['placement']=='rr' else 'cli-balanced-wave-aware',runtime_defaults={'llama_model_path':case['model_path'],
                'adb_serial':serial,'max_parallel_nodes':case['parallel']})
        overrides={n:dict(base,llama_device=d,ngl=0 if d=='none' else 99,hexagon_env=d=='HTP0') for n,d in placement.items()}
        llm={'runtime':'llama_cli','llama_cli':dict(base,llama_device='none',ngl=0),
             'llama_cli_overrides':overrides}
    elif case['runtime']=='ort_genai_cli':
        # No ORT-only truncation: all runtime paths receive the complete prompt.
        llm={'runtime':'ort_genai_cli','ort_genai':dict(common,runner_dir=case['runner_dir'],
              model_path=case['model_path'],jar=case['jar'],prompt_char_limit=0)}
    else:
        llm={'runtime':'genie_cli','genie':dict(common,bundle_dir=case['bundle_dir'],config=case['config'],backend='npu')}
    cfg={'max_parallel_nodes':case['parallel'],'llm':llm,'qnn_tools':{}}
    for n in workflow['nodes']:
        if n['kind'] in ('app','dnn'):
            tool=n['entrypoint']
            cfg['qnn_tools'][tool]={'mode':'json_cli','command':[adb,'-s',serial,'shell',
                shlex.join([remote+'/wfgen-tools',tool,'--state',remote+'/active.state.json'])],'timeout':60}
    return cfg


def generated(step):
    if not step.get('success'): return False
    raw=step.get('raw',{})
    if raw.get('mode')=='ort_genai_cli':
        return float(raw.get('generated_tokens') or 0)>0 and not raw.get('prompt_truncated')
    if raw.get('mode')=='genie_cli':
        return float(raw.get('profile_summary',{}).get('generated_tokens') or 0)>0
    return raw.get('target')=='phone'


def prepare(args):
    if args.output.exists(): raise ValueError('Output exists; choose a new directory to preserve prior attempts')
    if not args.output.is_relative_to(ROOT/'local_data'):
        raise ValueError('Experiment artifacts must stay in ignored local_data')
    existing=load_existing(args.existing_run,args.bundle,args.binary,args.runtime_source)
    for checkpoint in getattr(args,'reuse_run',[]):
        prior=json.loads((checkpoint/'manifest.json').read_text())
        if prior['status'] not in ('completed','interrupted'):
            raise ValueError('Reuse requires a stopped checkpoint')
        if prior['binary_sha256']!=sha(args.binary) or prior['workflow_sha256']!=sha(args.bundle/'workflows.json'):
            raise ValueError('Checkpoint binary/workflows mismatch')
        for relative,digest in prior['runtime_sha256'].items():
            if sha(args.runtime_source/relative)!=digest: raise ValueError('Checkpoint runtime mismatch')
        for path in list((args.bundle/'initial').glob('*.json'))+list((args.bundle/'expected').glob('*.json')):
            if sha(path)!=sha(checkpoint/'bundle'/path.relative_to(args.bundle)):
                raise ValueError('Checkpoint reference mismatch')
        source=checkpoint/('summary.json' if (checkpoint/'summary.json').exists() else 'progress.json')
        for row in json.loads(source.read_text())['rows']:
            if row.get('repeat'): raise ValueError('Repeated resource cases require an explicit continuation plan')
            if json.loads(Path(row['result_json']).read_text())['summary']!=row:
                raise ValueError('Checkpoint raw record differs from summary')
            key=(row['device'],row['model_id'],row['configuration'],row['source_index'])
            if key in existing: raise ValueError('Checkpoint duplicates an existing measurement')
            existing[key]=row
    catalog=load_catalog()
    catalog['phones']={DEVICE_A:'8 Elite',DEVICE_B:'8 Gen 3'}
    resource_candidates=[r for r in existing.values() if r['device']==DEVICE_B and
                         r['model_id']==QWEN and r['source_index']==179 and r['success']]
    if not resource_candidates: raise ValueError('No successful resource reference configuration')
    resource_winner=min(resource_candidates,key=lambda r:r['wall_ms'])
    args.output.mkdir(parents=True)
    save(args.output/'catalog.json',catalog)
    shutil.copy2(args.binary,args.output/'wfgen-tools')
    bundle=args.output/'bundle'
    shutil.copytree(args.bundle,bundle,ignore=lambda path,names:[n for n in names if
        (Path(path)==args.bundle and n not in ('initial','expected','workflows.json','provenance.json'))])
    package=args.output/'runtime/MOBench'
    shutil.copytree(args.runtime_source,package,ignore=shutil.ignore_patterns('__pycache__', '.git', 'local_data'))
    source_scripts=args.output/'source_scripts'
    source_scripts.mkdir()
    for name in ('run_selected10_supplement.py','run_comparison.py','scheduling.py','validate.py','prepare.py','selected10_matrix.py'):
        shutil.copy2(Path(__file__).parent/name,source_scripts/name)
    sys.path.insert(0,str(package.parent))
    from MOBench.workflow_execution.runtime import MobileWorkflowRuntime
    from MOBench.deployment_planning.core import llm_device_assignments
    workflows=json.loads((bundle/'workflows.json').read_text())['workflows']
    assert [w['metadata']['selection_index'] for w in workflows]==INDICES
    plans=[derive_plan(w,MobileWorkflowRuntime._topological_waves) for w in workflows]
    matrix=cases(args.suite,catalog,resource_winner['config'])
    devices=args.devices.split(',')
    if len(devices)!=len(set(devices)): raise ValueError('Duplicate phone serials')
    jobs=[];reused=[];all_jobs=[]
    for device in devices:
        if device not in (DEVICE_A,DEVICE_B): raise ValueError('Unknown phone serial')
        remote=f'/data/local/tmp/wfgen_native_tools/supplement_{os.getpid()}_{time.time_ns()}'
        for case in matrix:
            if device not in case.get('devices',devices): continue
            for workflow,added in plans:
                if workflow['metadata']['selection_index'] not in case.get('indices',INDICES): continue
                idx=workflow['metadata']['selection_index']
                prefix=args.output/device/case['id']/str(idx)
                cfg=runtime_config(case,workflow,device,args.adb,remote,llm_device_assignments)
                save(prefix.with_suffix('.runtime.json'),cfg)
                save(prefix.with_suffix('.workflow.json'),{'workflow':workflow,'added_dependencies':added})
                job={'device':device,'case':case,'source_index':idx,'remote':remote,
                     'prefix':str(prefix.relative_to(args.output))}
                all_jobs.append(job)
                old=existing.get((device,case['model_id'],case['configuration'],idx))
                if old and not case.get('repeat') and not args.no_reuse:
                    source=Path(old['result_json'])
                    old_prefix=source.parent/str(idx)
                    old_cfg=json.loads(old_prefix.with_suffix('.runtime.json').read_text())
                    old_workflow=json.loads(old_prefix.with_suffix('.workflow.json').read_text())['workflow']
                    if comparable_config(cfg)!=comparable_config(old_cfg) or workflow!=old_workflow:
                        raise ValueError('Existing execution protocol mismatch: '+str(source))
                    row=dict(old,config=case['id'],configuration=case['configuration'],
                             runtime=case['runtime'],placement=case['placement'],repeat=None,
                             source_config=old['config'],reused=True)
                    reused.append({'job':job,'row':row,'source_sha256':sha(source)})
                else: jobs.append(job)
    manifest={'status':'prepared_not_run','expected':len(jobs),'jobs':jobs,
              'all_jobs':all_jobs,'reused':reused,'total_with_reused':len(all_jobs),
              'suite':args.suite,'catalog':catalog,'adb':args.adb,
              'resource_tools_dir':str(args.resource_tools_dir),
              'resource_selection':{'rule':'Freeze fastest successful historical configuration before three fresh repeats',
                                    'source':resource_winner,'source_sha256':sha(Path(resource_winner['result_json']))},
              'selection':'For model/hardware panels choose fastest successful candidate per device/model/workflow among naive,wave2,wave3,max_rr,max_wave; no success => ERR.',
              'readiness':'Local plans validated; phone model/runtime availability NOT checked in prepare mode.',
              'runner_sha256':{p.name:sha(p) for p in source_scripts.glob('*.py')},
              'workflow_sha256':sha(bundle/'workflows.json'),'binary_sha256':sha(args.binary),
              'runtime_sha256':{str(p.relative_to(package)):sha(p) for p in package.rglob('*.py')},
              'parameters':{'max_tokens':64,'llm_timeout_s':240,'threads_llama':6,'prompt_truncation':False,
                            'warmup':'one per used llama backend or one per ORT/Genie case, outside measured timing',
                            'framework_parallelism':[1,2],'npu_only_parallelism':3,'fresh_resource_repeats':3},
              'scope':'Native phone business tools/state and phone LLMs, host DAG scheduler. No independent semantic-checker claim.',
              'failure_policy':'Keep every attempt, do not retry, ERR success/total in plots; missing measurements are N/A.',
              'framework_comparison':'Matched p=1 and p=2, same reference DAGs and full prompts. Package quantization/tokenizers differ; compare runtime/package paths, not kernels alone.'}
    manifest['input_sha256']={str(p.relative_to(args.output)):sha(p) for p in args.output.rglob('*')
                               if p.is_file() and '__pycache__' not in p.parts}
    save(args.output/'manifest.json',manifest)
    return manifest,MobileWorkflowRuntime


def preflight(args, manifest):
    """Read-only device checks, only reached after an explicit execution flag."""
    checks=[]
    for device in dict.fromkeys(j['device'] for j in manifest['jobs']):
        adb=[args.adb,'-s',device,'shell']
        def query(command):
            return subprocess.run(adb+[command],capture_output=True,text=True,timeout=30)
        identity=query('getprop ro.soc.model')
        expected={DEVICE_A:'SM8750',DEVICE_B:'SM8650'}[device]
        if identity.returncode or identity.stdout.strip()!=expected:
            raise RuntimeError('Phone unavailable or unexpected SoC: '+device)
        active=query('pidof llama-cli genie-t2t-run dalvikvm dalvikvm64 perfetto; true')
        if active.returncode or active.stdout.strip():
            raise RuntimeError('Phone unavailable or busy: '+device)
        dalvik=query('command -v dalvikvm')
        if dalvik.returncode or not dalvik.stdout.strip().startswith('/'):
            raise RuntimeError('Android Java runtime unavailable: '+device)
        required=set()
        for job in manifest['jobs']:
            if job['device']!=device: continue
            case=job['case']
            if case['runtime']=='llama_cli':
                required.update([('r',case['model_path']),('x','/data/local/tmp/llama.cpp/bin/llama-cli')])
            elif case['runtime']=='ort_genai_cli':
                required.update([('d',case['model_path']),('r',case['model_path']+'/genai_config.json'),
                                 ('r',case['runner_dir']+'/'+case['jar']),('x',dalvik.stdout.strip())])
            else:
                config='genie_config_sm8750.json' if device==DEVICE_A else 'genie_config.json'
                required.update([('x',case['bundle_dir']+'/genie-t2t-run'),('r',case['bundle_dir']+'/'+config)])
        for kind,path in sorted(required):
            check=query(f'test -{kind} '+shlex.quote(path))
            checks.append({'device':device,'path':path,'check':kind,'available':check.returncode==0})
    save(args.output/'preflight.json',{'checks':checks,'ready':all(c['available'] for c in checks)})
    if not all(c['available'] for c in checks):
        raise RuntimeError('Missing phone packages; see preflight.json. No experiments started.')


def execute(args,manifest,Runtime):
    from MOBench.tools.llm_tau import LlamaCliClient
    from MOBench.resource_monitoring import PerfettoResourceMonitor,ResourceMonitorConfig
    import MOBench.resource_monitoring.perfetto_monitor as monitoring
    rows=[]
    manifest['status']='running';save(args.output/'manifest.json',manifest)
    for device in dict.fromkeys(j['device'] for j in manifest['jobs']):
        jobs=[j for j in manifest['jobs'] if j['device']==device]
        remote=jobs[0]['remote'];owner=remote.rsplit('/',1)[-1]
        adb=[args.adb,'-s',device]
        def shell(command,check=True):
            return subprocess.run(adb+['shell',command],capture_output=True,text=True,timeout=60,check=check)
        class OwnedRuntime(Runtime):
            def _adb_shell(self,cfg,command,**kw):
                return super()._adb_shell(cfg,f'export WFGEN_OWNER_TOKEN={shlex.quote(owner)}; '+command,**kw)
        if shell('pidof llama-cli genie-t2t-run dalvikvm dalvikvm64 perfetto; true').stdout.strip():
            raise RuntimeError('Phone has an active inference or trace process')
        shell('mkdir -p '+shlex.quote(remote+'/prompts'))
        subprocess.run(adb+['push',str(args.binary),remote+'/wfgen-tools'],capture_output=True,check=True,timeout=60)
        shell('chmod 755 '+shlex.quote(remote+'/wfgen-tools'))
        assert shell('sha256sum '+shlex.quote(remote+'/wfgen-tools')).stdout.split()[0]==manifest['binary_sha256']
        for domain in ('retail','airline'):
            path=args.output/'bundle/initial'/f'{domain}.json'
            subprocess.run(adb+['push',str(path),remote+f'/{domain}.initial.json'],capture_output=True,check=True,timeout=60)
            assert shell('sha256sum '+shlex.quote(remote+f'/{domain}.initial.json')).stdout.split()[0]==sha(path)
        save(args.output/device/'device_initial.json',{'output':shell('getprop ro.build.fingerprint; dumpsys battery').stdout})
        def cleanup():
            for pid in shell('pidof llama-cli genie-t2t-run dalvikvm dalvikvm64; true',False).stdout.split():
                if not pid.isdigit(): continue
                cmd=shell(f'cat /proc/{pid}/cmdline',False).stdout
                env=shell(f'cat /proc/{pid}/environ',False).stdout
                if remote+'/prompts/' in cmd or 'WFGEN_OWNER_TOKEN='+owner in env.split('\0'):
                    shell('kill -TERM '+pid,False)
                    # Kill only a process whose ownership remains verifiable.
                    time.sleep(.1)
                    remaining_cmd=shell(f'cat /proc/{pid}/cmdline',False).stdout
                    remaining_env=shell(f'cat /proc/{pid}/environ',False).stdout
                    if remote+'/prompts/' in remaining_cmd or 'WFGEN_OWNER_TOKEN='+owner in remaining_env.split('\0'):
                        shell('kill -KILL '+pid,False)
        for case_id in dict.fromkeys(j['case']['id'] for j in jobs):
            phone_free=int(shell("df -k /data/local/tmp | tail -1").stdout.split()[3])*1024
            if phone_free<512*1024**2 or shutil.disk_usage(args.output).free<2*1024**3:
                raise RuntimeError('Insufficient free disk space before next case; preserve all existing records')
            batch=[j for j in jobs if j['case']['id']==case_id]
            folder=args.output/device/case_id
            first=args.output/batch[0]['prefix']
            cfg=json.loads(first.with_suffix('.runtime.json').read_text())
            w=json.loads(first.with_suffix('.workflow.json').read_text())['workflow']
            if cfg['llm']['runtime']=='llama_cli':
                configs=[]
                for job in batch:
                    config_path=(args.output/job['prefix']).with_suffix('.runtime.json')
                    configs.extend(json.loads(config_path.read_text())['llm']['llama_cli_overrides'].values())
                unique={c['llama_device']:c for c in configs}
                warm=[]
                for backend,c in unique.items():
                    try: warm.append({'backend':backend,'result':LlamaCliClient.from_config(c).warmup()})
                    except Exception: warm.append({'backend':backend,'error':traceback.format_exc()})
                    finally: cleanup()
            else:
                node=copy.deepcopy(next(n for n in w['nodes'] if n['kind']=='llm'))
                try: warm=[OwnedRuntime(cfg)._run_llm_node(node,w,{})]
                except Exception: warm=[{'error':traceback.format_exc()}]
                finally: cleanup()
            save(folder/'warmup.json',warm)
            monitoring.REMOTE_TRACE_FILENAME=owner+'_'+case_id+'.perfetto-trace'
            monitoring.REMOTE_CONFIG_FILENAME=owner+'_'+case_id+'.pbtxt'
            monitor=PerfettoResourceMonitor(ResourceMonitorConfig(adb_path=Path(args.adb),device_serial=device,
                      output_dir=folder/'resources',tools_dir=Path(manifest['resource_tools_dir']),buffer_mb=128,required=True))
            records=[]
            try:
                save(folder/'monitor_start.json',monitor.start())
                for job in batch:
                    prefix=args.output/job['prefix'];idx=job['source_index']
                    w=json.loads(prefix.with_suffix('.workflow.json').read_text())['workflow']
                    cfg=json.loads(prefix.with_suffix('.runtime.json').read_text())
                    domain='airline' if '_airline_' in w['workflow_id'] else 'retail'
                    row={'device':device,'model_id':job['case']['model_id'],'config':case_id,'source_index':idx,
                         'workflow_id':w['workflow_id'],'runtime':job['case']['runtime'],
                         'max_parallel_nodes':job['case']['parallel'],'configuration':job['case']['configuration'],
                         'placement':job['case']['placement'],'repeat':job['case'].get('repeat'),'reused':False,'status':'failed','success':False,
                         'wall_ms':None,'failure_reason':'','result_json':str(prefix.with_suffix('.result.json'))}
                    result=None;started=time.time();interruption=None
                    try:
                        shell('cp '+shlex.quote(remote+f'/{domain}.initial.json')+' '+shlex.quote(remote+'/active.state.json'))
                        assert shell('sha256sum '+shlex.quote(remote+'/active.state.json')).stdout.split()[0]==sha(args.output/'bundle/initial'/f'{domain}.json')
                        started=time.time();wall_start=time.perf_counter()
                        result=OwnedRuntime(cfg).execute_workflow(w)
                        row['wall_ms']=result['total_wall_ms']
                        row['host_execute_call_ms']=(time.perf_counter()-wall_start)*1000
                        actual=json.loads(shell('cat '+shlex.quote(remote+'/active.state.json')).stdout)
                        save(prefix.with_suffix('.state.json'),actual)
                        refs={s['node_id']:s['reference'] for s in json.loads((args.output/'bundle/expected'/f'{idx}.steps.json').read_text())}
                        tools=[s for s in result['steps'] if s['kind'] in ('app','dnn')]
                        llms=[s for s in result['steps'] if s['kind']=='llm']
                        row['state_matches_reference']=equal(actual,json.loads((args.output/'bundle/expected'/f'{idx}.state.json').read_text()))
                        row['tool_outputs_match_reference']=Counter(s['node_id'] for s in tools)==Counter(refs.keys()) and all(
                            s['success'] and s['raw'].get('response',{}).get('raw',{}).get('platform')=='android' and
                            equal(s['raw'].get('response',{}).get('raw',{}).get('result'),refs[s['node_id']].get('result')) for s in tools)
                        row['generation_valid']=len(llms)==sum(n['kind']=='llm' for n in w['nodes']) and all(generated(s) for s in llms)
                        row['success']=result['status']=='completed' and all(row[k] for k in ('state_matches_reference','tool_outputs_match_reference','generation_valid'))
                        row['status']='completed' if row['success'] else 'failed'
                        if not row['success']: row['failure_reason']='Execution, generation, tool output, or final-state validation failed; see raw result.'
                    except BaseException as exc:
                        row['failure_reason']=traceback.format_exc()
                        if not isinstance(exc,Exception): interruption=exc
                    finally:
                        finished=time.time()
                        if row['wall_ms'] is None: row['wall_ms']=(finished-started)*1000
                        try: cleanup()
                        except Exception:
                            row['cleanup_error']=traceback.format_exc();row['success']=False;row['status']='failed'
                    save(prefix.with_suffix('.result.json'),{'execution':result,'summary':row})
                    rows.append(row)
                    records.append({'workflow_id':w['workflow_id'],'workflow_index':idx,'status':row['status'],
                                    'started_epoch_s':started,'finished_epoch_s':finished,'result_json':row['result_json']})
                    save(args.output/'progress.json',{'complete':False,'expected':manifest['expected'],'rows':rows})
                    print('DONE',len(rows),'/',manifest['expected'],device,case_id,idx,row['status'],flush=True)
                    if interruption is not None: raise interruption
            finally:
                info=monitor.stop(records)
                save(folder/'monitor_finish.json',info)
                # Release only this run's trace after verifying the retained local copy.
                remote_trace=monitor.remote_trace_path
                local_trace=monitor.trace_path
                if (info.get('trace_pull',{}).get('returncode')==0 and local_trace.exists()
                        and local_trace.stat().st_size>0 and
                        Path(remote_trace).name==owner+'_'+case_id+'.perfetto-trace'):
                    remote_hash=shell('sha256sum '+shlex.quote(remote_trace)).stdout.split()[0]
                    if sha(local_trace)!=remote_hash:
                        raise RuntimeError('Trace transfer checksum mismatch; remote copy retained')
                    shell('rm -- '+shlex.quote(remote_trace))
                    save(folder/'trace_archive.json',{'sha256':remote_hash,'local_path':str(local_trace),
                         'remote_path':remote_trace,'remote_removed_after_verified_transfer':True})
        save(args.output/device/'device_final.json',{'output':shell('dumpsys battery; pidof llama-cli genie-t2t-run dalvikvm perfetto; true').stdout})
    assert len(rows)==manifest['expected']
    save(args.output/'summary.json',{'complete':True,'expected':manifest['expected'],'rows':rows})
    manifest['status']='completed';manifest['successes']=sum(r['success'] for r in rows)
    save(args.output/'manifest.json',manifest)


def main():
    global DEVICE_A,DEVICE_B
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--suite',choices=SUITES,default='all')
    ap.add_argument('--devices',help='Comma-separated actual serials; defaults to both configured device profiles.')
    ap.add_argument('--device-a-serial',default=DEVICE_A,help='Actual serial of the SM8750 / 8 Elite device.')
    ap.add_argument('--device-b-serial',default=DEVICE_B,help='Actual serial of the SM8650 / 8 Gen 3 device.')
    ap.add_argument('--resource-tools-dir',type=Path,default=ROOT/'local_data/perfetto_tools')
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--bundle',type=Path,default=ROOT/'local_data/phone_tools_v1')
    ap.add_argument('--runtime-source',type=Path,default=DEFAULT_RUNTIME)
    ap.add_argument('--binary',type=Path,default=ROOT/'local_data/selected10_four_methods_20260928/build-android/wfgen-tools')
    ap.add_argument('--adb',default='adb')
    ap.add_argument('--existing-run',type=Path,default=ROOT/'local_data/selected10_four_methods_20260928')
    ap.add_argument('--no-reuse',action='store_true',help='Explicitly measure a new full cohort instead of reusing old records.')
    ap.add_argument('--reuse-run',type=Path,action='append',default=[],help='Reuse completed records from a stopped checkpoint; never retries a failed record.')
    ap.add_argument('--execute-prepared',action='store_true',help='Execute the exact untouched prepared output; refuses started/completed runs.')
    ap.add_argument('--execute',action='store_true',help='Actually use phones; without this flag only prepare local plans.')
    args=ap.parse_args();args.output=args.output.resolve();args.bundle=args.bundle.resolve()
    DEVICE_A,DEVICE_B=args.device_a_serial,args.device_b_serial
    if DEVICE_A==DEVICE_B: ap.error('Device profile serials must differ')
    args.devices=args.devices or (DEVICE_A+','+DEVICE_B)
    import selected10_matrix
    selected10_matrix.DEVICE_B=DEVICE_B
    os.environ.setdefault('NO_PROXY','localhost,127.0.0.1,::1');os.environ.setdefault('no_proxy',os.environ['NO_PROXY'])
    if args.execute_prepared:
        if args.execute or args.no_reuse: ap.error('Do not combine execute-prepared with execute/no-reuse')
        manifest=json.loads((args.output/'manifest.json').read_text())
        if manifest['status']!='prepared_not_run': raise ValueError('Run has already started; no automatic restart/retry')
        for name,digest in manifest['runner_sha256'].items():
            if sha(Path(__file__).parent/name)!=digest: raise ValueError('Runner changed since preparation: '+name)
        for relative,digest in manifest['input_sha256'].items():
            if sha(args.output/relative)!=digest: raise ValueError('Prepared input changed: '+relative)
        for item in manifest['reused']:
            if sha(Path(item['row']['result_json']))!=item['source_sha256']: raise ValueError('Reused result changed')
        args.binary=args.output/'wfgen-tools';args.adb=manifest['adb']
        sys.path.insert(0,str(args.output/'runtime'))
        from MOBench.workflow_execution.runtime import MobileWorkflowRuntime as Runtime
    else:
        manifest,Runtime=prepare(args)
        args.binary=args.output/'wfgen-tools'
    if args.execute or args.execute_prepared:
        preflight(args,manifest)
        try: execute(args,manifest,Runtime)
        except BaseException:
            manifest['status']='interrupted';manifest['error']=traceback.format_exc()
            save(args.output/'manifest.json',manifest);raise
    else: print(json.dumps({'status':'prepared_not_run','new_attempts':manifest['expected'],'reused':len(manifest['reused']),'total':manifest['total_with_reused'],'output':str(args.output)}))


if __name__=='__main__': main()
