"""Frozen original executor with guards outside workflow timing; no retries."""
import inspect,json,sys,time
from pathlib import Path
if '--help' in sys.argv or '-h' in sys.argv:
    sys.path.insert(0,str(Path(__file__).resolve().parent/'mobile_tools/scripts'))
else:
    RUN=Path(sys.argv[sys.argv.index('--output')+1]).resolve()
    sys.path.insert(0,str(RUN/'source_scripts'))
import run_selected10_supplement as base
source=inspect.getsource(base.execute)
needle='                for job in batch:\n                    prefix='
replacement="""                for job in batch:
                    if (args.output.parent/'pause_request.json').exists():
                        raise RuntimeError('Pause requested before next workflow')
                    identity=shell('getprop ro.serialno').stdout.strip()
                    if identity != device: raise RuntimeError('Disconnected or wrong device before next workflow')
                    boot_before=shell('cat /proc/sys/kernel/random/boot_id').stdout.strip()
                    prefix="""
assert source.count(needle)==1;source=source.replace(needle,replacement)
needle="                        started=time.time();wall_start=time.perf_counter()"
replacement="""                        save(args.output/'active_workflow.json',dict(device=device,config=case_id,source_index=idx,
                             result_json=row['result_json'],started_epoch=time.time(),boot_id=boot_before))
                        print('START',device,case_id,idx,flush=True)
                        row['boot_id_before']=boot_before
                        started=time.time();wall_start=time.perf_counter()"""
assert source.count(needle)==1;source=source.replace(needle,replacement)
needle="                        actual=json.loads(shell('cat '+shlex.quote(remote+'/active.state.json')).stdout)"
replacement="""                        identity_after=shell('getprop ro.serialno').stdout.strip()
                        row['boot_id_after']=shell('cat /proc/sys/kernel/random/boot_id').stdout.strip()
                        if identity_after!=device or row['boot_id_after']!=boot_before:
                            raise RuntimeError('Device disconnected or rebooted during workflow')
                        actual=json.loads(shell('cat '+shlex.quote(remote+'/active.state.json')).stdout)"""
assert source.count(needle)==1;source=source.replace(needle,replacement)
needle="                    save(prefix.with_suffix('.result.json'),{'execution':result,'summary':row})"
replacement="""                    errors=' '.join(str(s.get('raw',{})) for s in (result or {}).get('steps',[]) if not s.get('success'))+' '+row['failure_reason']
                    if not row['success']:
                        if 'reboot' in errors.lower(): row['failure_category']='device_reboot_or_disconnect'
                        elif any(s in errors.lower() for s in ['device not found','device offline','disconnected','transport error']): row['failure_category']='device_disconnect'
                        elif any(s.get('raw',{}).get('returncode') in (124,137) for s in (result or {}).get('steps',[])): row['failure_category']='inference_timeout'
                        else: row['failure_category']='backend_or_validation_error'
                    row['partial_wall_ms']=row['wall_ms'] if not row['success'] else None
                    if not row['success']: row['wall_ms']=None
                    save(prefix.with_suffix('.result.json'),{'execution':result,'summary':row})"""
assert source.count(needle)==1;source=source.replace(needle,replacement)
needle="                    if interruption is not None: raise interruption"
replacement="""                    if row.get('failure_category') in ('device_reboot_or_disconnect','device_disconnect'):
                        raise RuntimeError('Connection anomaly: current workflow retained ERR, queue stopped')
                    if idx in (120,48) and not row['success']:
                        raise RuntimeError('Pilot workflow failed; preserve ERR and stop before remaining cohort')
                    if interruption is not None: raise interruption"""
assert source.count(needle)==1;source=source.replace(needle,replacement)
exec(compile(source,__file__,'exec'),base.__dict__)
if __name__=='__main__':base.main()
