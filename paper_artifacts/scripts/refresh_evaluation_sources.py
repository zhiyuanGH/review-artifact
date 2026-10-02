"""Refresh Evaluation data only, using the existing paper figure renderers.

--refresh reads a separately supplied audited two-phone cohort. Without it, a
separately supplied evaluation_sources.json snapshot supports figure redrawing
without local raw measurements. Neither dataset is included in this code archive.
No phone calls, paper text edits, style edits, or experiment execution.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from statistics import mean

import generate_studies_execution_figures as execution
import generate_studies_characterization_figures as characterization
import generate_selected10_followup_figures as followup

ROOT=Path(__file__).resolve().parents[1]
SNAPSHOT=ROOT/'data/evaluation_sources.json'
DEFAULT_RUN=ROOT/'local_data/selected10_paper_supplement_parallel_20260930/combined'
EX=ROOT/'figures/studies/execution'
CH=ROOT/'figures/studies/characterization'
DEVICES=['elite','gen3']
# Preserve the current paper's series and aggregation scope.
MODELS=['qwen3_0_6b_q4_k_m','llama_3_2_1b_instruct_q4_0','qwen1_5_1_8b_chat_q4_0']
CANDIDATES=['naive','wave2','wave3','max_rr','max_wave']
# Framework characterization also has a measured serial wave-aware candidate.
FRAMEWORK_LLAMA_CANDIDATES=['naive','wave1','wave2','wave3','max_rr','max_wave']
# Model Characterization's eight packages and order in the July 1 paper.
# Keep the three-package execution/hardware scope above unchanged.
MODEL_REFERENCE_COMMIT='frozen_eight_model_package_order'
MODEL_CHARACTERIZATION_MODELS=['qwen3_0_6b_q4_k_m','qwen3_0_6b_q8_0',
 'qwen3_0_6b_q6_k','qwen3_0_6b_bf16','llama_3_2_1b_instruct_q4_0',
 'qwen1_5_1_8b_chat_q4_0','llama_3_2_3b_instruct_q4_0','hunyuan_0_5b_instruct_q6_k']

def read(p):return json.loads(Path(p).read_text())
def write(p,value):Path(p).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def source(p):
 p=Path(p);return {'path':str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p),'sha256':digest(p)}
def label(index):return ('A' if index in (186,213,239) else 'R')+str(index)

def build(run:Path):
 manifest=read(run/'manifest.json');audit=read(run/'final_validation.json');summary=read(run/'analysis/summary.json')
 rows=read(run/'summary.json')['rows']+[x['row'] for x in manifest['reused']]
 assert audit['passed'] and len(rows)==1295 and len({(r['device'],r['config'],r['source_index']) for r in rows})==1295
 for path,sha in audit['all_original_sha256'].items():assert digest(path)==sha,path
 indices=manifest['catalog']['workflow_indices']
 assert indices==[120,7,33,100,186,213,239,48,152,179]
 meta={j['case']['id']:j['case'] for j in manifest['all_jobs']}
 by={(r['device'],r['model_id'],meta[r['config']]['configuration'],r['source_index']):r for r in rows if not meta[r['config']].get('repeat')}
 selected={(r['device'],r['model_id'],r['source_index']):r for r in summary['odabench_selection']}
 for (device,model,index),s in selected.items():
  options=[by[device,model,c,index] for c in CANDIDATES]
  good=[r for r in options if r['success']]
  assert s['candidates']==5
  if good:assert s['winner'] in good and s['wall_ms']==min(r['wall_ms'] for r in good)
 sources={name:source(run/name) for name in ['manifest.json','summary.json','analysis/summary.json','transport_provenance.json','final_validation.json']}
 latency=read(EX/'execution_latency_selected10.json');latency['dataset']=str(run.relative_to(ROOT))
 selections=[];methods={k:[] for k in ['naive','odabench','max_rr','max_wave']}
 for d in DEVICES:
  for m in MODELS:
   for i in indices:
    winner=selected[d,m,i]['winner'];assert winner and winner['success']
    selections.append(dict(device=d,model_id=m,source_index=i,workflow_id=winner['workflow_id'],configuration=meta[winner['config']]['configuration'],p=winner['max_parallel_nodes'],wall_ms=winner['wall_ms'],successful=1,total=1,candidate_count=5,successful_candidates=selected[d,m,i]['successful_candidates'],result_json=winner['result_json']))
    methods['odabench'].append(winner)
    for method in ['naive','max_rr','max_wave']:methods[method].append(by[d,m,method,i])
 for out in latency['rows']:
  i=out['source_index']
  for method,items in methods.items():
   group=[r for r in items if r['source_index']==i];assert len(group)==6
   out.update({method+'_sample_count':6,method+'_successful':sum(r['success'] for r in group),method+'_case_values_s':[r['wall_ms']/1000 for r in group],method+'_case_successes':[r['success'] for r in group],method+'_attempt_s':mean(r['wall_ms']/1000 for r in group),method+'_s':mean(r['wall_ms']/1000 for r in group) if all(r['success'] for r in group) else None})
  out['speedup_vs_naive_x']=out['naive_s']/out['odabench_s'];out['reduction_vs_naive_pct']=100*(1-out['odabench_s']/out['naive_s'])
 latency['latency'].update(rows=latency['rows'],sources=sources,selection=selections,success_counts={k:{'successful':sum(r['success'] for r in items),'total':len(items)} for k,items in methods.items()})
 for method in ['naive','odabench']:
  latency[method]=execution.summarize_series(latency['rows'],method);latency['latency'][method]=latency[method]
 latency['improvement']={'speedup_naive_vs_odabench_x':latency['naive']['total_s']/latency['odabench']['total_s'],'latency_reduction_naive_vs_odabench_pct':100*(1-latency['odabench']['total_s']/latency['naive']['total_s'])}
 data=read(EX/'selected10_followup_provenance.json');data.update(dataset=str(run.relative_to(ROOT)),source_sha256=sources,selection=selections,pending_measurements=[])
 source_rows=[dict(device=s['device'],model_id=s['model_id'],framework_id='llama_cpp_selected',wall_s=s['wall_ms']/1000,workflow_index=indices.index(s['source_index']),workflow_id=s['workflow_id'],source_index=s['source_index'],selected_configuration=s['configuration']) for s in selections]
 data['source_rows']=source_rows;data['hardware_by_workflow']=characterization.build_hardware_by_workflow(source_rows,DEVICES);data['model_by_workflow']=characterization.build_model_by_workflow(source_rows,MODELS)
 for out in data['structure_ablation']['rows']:
  d,m=out['device'],out['model_id'];ours=[selected[d,m,i] for i in indices];fixed=[by[d,m,'wave3',i] for i in indices]
  good=sum(r['success'] for r in fixed);v=sum(r['wall_ms'] for r in fixed)/60000;vours=sum(r['wall_ms'] for r in ours)/60000
  out.update(odabench_total_wall_min=vours,blind_p3_attempt_wall_min=v,blind_p3_total_wall_min=v if good==10 else None,blind_p3_successes=good,total=10,blind_p3_clean=good==10,blind_p3_over_odabench=v/vours if good==10 else None)
 clean=[r for r in data['structure_ablation']['rows'] if r['blind_p3_clean']];x=sum(r['odabench_total_wall_min'] for r in clean);y=sum(r['blind_p3_total_wall_min'] for r in clean)
 data['structure_ablation']['summary'].update(clean_case_count=len(clean),odabench_total_min=x,blind_p3_total_min=y,blind_p3_over_odabench=y/x,latency_reduction_vs_blind_p3_pct=100*(1-x/y),max_case_over_odabench=max(r['blind_p3_over_odabench'] for r in clean))
 # The paper's resource panel remains one Gen 3 observed run, not a new plot
 # type or a pooled/normalized timeline. Repeat 1 is fixed before inspection.
 timeline=dict(workflow_index=179,workflow_label='R179',device='gen3',model_id=MODELS[2],mode='single_observed_run',repeat_selection='Predeclared repeat 1; no fastest-repeat selection or uncertainty bands.',gpu_sample_policy=followup.GPU_SAMPLE_POLICY,methods={},source_sha256={})
 for method,role in [('naive','naive'),('odabench','odabench')]:
  case=next(r for r in summary['resource_runs'] if r['figure']=='Fig.13' and r['role']==role and r['repeat']==1)
  row=next(r for r in rows if r['device']=='gen3' and r['config'].endswith('Fig13_'+role+'_repeat1'))
  raw=read(row['result_json']);assert row['success'];started=datetime.fromisoformat(raw['execution']['started_at_utc'].replace('Z','+00:00')).timestamp();ended=datetime.fromisoformat(raw['execution']['finished_at_utc'].replace('Z','+00:00')).timestamp()
  points,zero_counter_count=followup.read_resource_samples(case['resource_samples'],started,ended)
  assert points
  timeline['methods'][method]=dict(configuration=meta[row['config']]['configuration'],repeat=1,duration_s=row['wall_ms']/1000,sample_count=len(points),samples=points,gpu_zero_counter_sample_count=zero_counter_count,gpu_missing_sample_count=sum(p['gpu_pct'] is None for p in points),runs=[{'samples':points}],averages={k:mean(p[k] for p in points if p[k] is not None) if any(p[k] is not None for p in points) else None for k in ['cpu_pct','gpu_pct','mem_gb']},result_json=row['result_json'])
  for p in [case['resource_samples'],case['resource_by_workflow'],row['result_json']]:timeline['source_sha256'][str(Path(p).relative_to(ROOT))]=digest(p)
 data['timeline']=timeline
 # Preserve two device subplots, Oracle/w/o decider series and existing ERRs.
 decider=[]
 for d in execution.QWEN15_DEVICE_ORDER:
  for pos,i in enumerate(indices):
   for method,r in [('odabench',selected[d,MODELS[2],i]['winner']),('wo_decider',by[d,MODELS[2],'npu_p3',i])]:
    decider.append(dict(workflow_index=pos,source_index=i,workflow_id=r['workflow_id'],workflow_label=label(i),device=d,device_label=execution.DEVICE_LABELS[d],model_id=MODELS[2],method=method,latency_s=r['wall_ms']/1000 if r['success'] else None,partial_wall_s=r['wall_ms']/1000,status='completed' if r['success'] else 'failed',marker='' if r['success'] else 'ERR 0/1',successes=int(r['success']),expected=1,result_json=r['result_json'],failure_reason=r.get('failure_reason','')))
 decider_data={'rows':decider,'summary':{'workflow_device_count':20,'wo_decider_failed_workflow_count':sum(r['status']!='completed' for r in decider if r['method']=='wo_decider'),'selection':'Same five-candidate fastest-success policy for Oracle; NPU-only p=3 for w/o decider.'}}
 char=read(CH/'characterization_comparison_provenance.json');char.update(model_order=MODELS,device_order=DEVICES,hardware_by_workflow=data['hardware_by_workflow'],model_by_workflow=data['model_by_workflow'],model_source_rows=source_rows,sources=sources)
 # Each framework uses exactly the same two devices x p=1,p=2. Preserve the
 # existing three-series single panel, averaging matched cells equally.
 frames=[];matched=[]
 for pos,i in enumerate(indices):
  values=[]
  for fid,runtime in [('genie_npu_phi35','genie_cli'),('ort_genai_cpu_phi35','ort_genai_cli'),('llama_cpp','llama_cli')]:
   group=[r for r in rows if r['source_index']==i and 'Fig.15' in meta[r['config']].get('figures',[]) and meta[r['config']]['runtime']==runtime]
   assert len(group)==4 and {(r['device'],r['max_parallel_nodes']) for r in group}=={(d,p) for d in DEVICES for p in [1,2]}
   successes=sum(r['success'] for r in group);average=mean(r['wall_ms']/1000 for r in group) if successes==4 else None
   values.append(dict(framework_id=fid,avg_wall_s=average,successes=successes,expected=4,sample_count=4,device_count=2,model_count=1,combo_count=2,marker='' if successes==4 else f'ERR {successes}/4',matched_cells=[dict(device=r['device'],parallel=r['max_parallel_nodes'],success=r['success'],wall_ms=r['wall_ms'],result_json=r['result_json']) for r in group]))
   for p in [1,2]:
    g=[r for r in group if r['max_parallel_nodes']==p];matched.append(dict(source_index=i,framework_id=fid,parallel=p,successes=sum(r['success'] for r in g),expected=2,latency_s=mean(r['wall_ms']/1000 for r in g) if all(r['success'] for r in g) else None))
  frames.append(dict(workflow_index=pos,workflow_label=label(i),values=values))
 char.update(framework_by_workflow=frames,framework_matched_parallelism=matched,datasets_by_figure={k:str(run.relative_to(ROOT)) for k in ['hardware','model','framework']})
 return dict(schema_version=1,sources=sources,scope={'models':MODELS,'devices':DEVICES,'resource_device':'gen3','resource_repeat':1,'unchanged':'All original rendering styles, series, plot dimensions and panel layouts. Generation/fidelity remain on their original non-phone sources.','transport':'USB and wireless history retained; host clocks include ADB communication overhead.'},latency=latency,followup=data,decider_ablation=decider_data,characterization=char)

def apply_ort_correction(data,run:Path):
 """Replace only Fig.15's ORT cells with the audited once-per-condition cohort."""
 summary=read(run/'summary.json');audit=read(run/'final_validation.json')
 assert summary['complete'] and audit['passed'] and summary['repeats']==1
 rows=summary['rows'];assert len(rows)==40
 assert len({(r['device'],r['max_parallel_nodes'],r['source_index']) for r in rows})==40
 for paths in [audit['original_raw_sha256'],audit['original_input_sha256'],audit['frozen_input_sha256'],audit['new_raw_sha256']]:
  for path,sha in paths.items():assert digest(path)==sha,path
 for row in rows:
  assert read(row['result_json'])['summary']==row
 result=copy.deepcopy(data);char=result['characterization'];fid='ort_genai_cpu_phi35'
 previous=char.get('framework_ort_correction',{})
 original=previous.get('original_failed_cells') or [dict(workflow_label=f['workflow_label'],**copy.deepcopy(v)) for f in char['framework_by_workflow'] for v in f['values'] if v['framework_id']==fid]
 for frame in char['framework_by_workflow']:
  i=int(frame['workflow_label'][1:]);group=[r for r in rows if r['source_index']==i]
  assert len(group)==4 and {(r['device'],r['max_parallel_nodes']) for r in group}=={(d,p) for d in DEVICES for p in [1,2]}
  point=next(v for v in frame['values'] if v['framework_id']==fid)
  successes=sum(r['success'] for r in group)
  point.update(avg_wall_s=mean(r['wall_ms']/1000 for r in group) if successes==4 else None,
   successes=successes,expected=4,sample_count=4,marker='' if successes==4 else f'ERR {successes}/4',
   matched_cells=[dict(device=r['device'],parallel=r['max_parallel_nodes'],success=r['success'],wall_ms=r['wall_ms'],result_json=r['result_json']) for r in group])
 for point in char['framework_matched_parallelism']:
  if point['framework_id']!=fid:continue
  group=[r for r in rows if r['source_index']==point['source_index'] and r['max_parallel_nodes']==point['parallel']]
  assert len(group)==2
  successes=sum(r['success'] for r in group)
  point.update(successes=successes,expected=2,latency_s=mean(r['wall_ms']/1000 for r in group) if successes==2 else None)
 char['framework_ort_correction']={
  'run':str(run.relative_to(ROOT)),'sources':{name:source(run/name) for name in ['summary.json','final_validation.json','continuation_audit.json','device_final_audit.json']},
  'framework_id':fid,'repeats':1,'successes':summary['successes'],'expected':40,
  'aggregation':'Equal-weight mean of the same two devices x p1/p2; one observation per cell. No successful-only mean for failed groups.',
  'change':'ORT Java total max_length = prompt_tokens + 64; full prompt, generation budget and native libraries retained.',
  'original_failed_cells':original,'original_40_ERR_retained':True,
  'transport':'Corrected ORT cohort uses USB. Other framework cells retain their original USB/wireless provenance; host timing includes ADB overhead.',
  'resource_caveat':summary['resource_caveat']}
 char['datasets_by_figure']['framework']={'original_cohort':str(DEFAULT_RUN.relative_to(ROOT)),'ort_corrected_cohort':str(run.relative_to(ROOT))}
 return result

def apply_framework_selection(data,run:Path):
 """Select one measured successful deployment per framework/phone/workflow."""
 manifest=read(run/'manifest.json');audit=read(run/'final_validation.json')
 assert audit['passed']
 rows=read(run/'summary.json')['rows']+[item['row'] for item in manifest['reused']]
 meta={job['case']['id']:job['case'] for job in manifest['all_jobs']}
 indices=manifest['catalog']['workflow_indices']
 result=copy.deepcopy(data);char=result['characterization']
 correction=char.get('framework_ort_correction')
 if not correction:raise ValueError('Framework selection requires the audited corrected ORT cohort')
 ort_run=ROOT/correction['run'];ort=read(ort_run/'summary.json');ort_audit=read(ort_run/'final_validation.json')
 assert ort['complete'] and ort['repeats']==1 and ort_audit['passed']
 for device_folder in ['elite','gen3']:
  assert read(ort_run/device_folder/'manifest.json')['workflow_sha256']==manifest['workflow_sha256']
 cohorts={
  'llama_cpp':[r for r in rows if r['model_id']=='phi_3_5_mini_instruct_q4_k_m' and
   meta[r['config']]['configuration'] in FRAMEWORK_LLAMA_CANDIDATES and not meta[r['config']].get('repeat')],
  'genie_npu_phi35':[r for r in rows if meta[r['config']]['runtime']=='genie_cli' and 'Fig.15' in meta[r['config']].get('figures',[])],
  'ort_genai_cpu_phi35':ort['rows']}
 expected_configs={'llama_cpp':set(FRAMEWORK_LLAMA_CANDIDATES),
  'genie_npu_phi35':{'genie_p1','genie_p2'},'ort_genai_cpu_phi35':{'ort_p1','ort_p2'}}
 original_hashes=audit['all_original_sha256'];ort_hashes=ort_audit['new_raw_sha256']
 selected=[];candidate_total=0;candidate_successes=0
 previous=char.get('framework_offline_selection',{})
 original_frames=previous.get('previous_matched_average_by_workflow') or copy.deepcopy(char['framework_by_workflow'])
 for fid,items in cohorts.items():
  expected=len(DEVICES)*len(indices)*len(expected_configs[fid])
  assert len(items)==expected and len({(r['device'],r['config'],r['source_index']) for r in items})==expected
  hashes=ort_hashes if fid=='ort_genai_cpu_phi35' else original_hashes
  for r in items:
   assert digest(r['result_json'])==hashes[r['result_json']]
   assert read(r['result_json'])['summary']==r
  candidate_total+=len(items);candidate_successes+=sum(r['success'] for r in items)
  for pos,i in enumerate(indices):
   frame=char['framework_by_workflow'][pos];assert frame['workflow_label']==label(i)
   point=next(v for v in frame['values'] if v['framework_id']==fid)
   winners=[];cells=[]
   for device in DEVICES:
    group=[r for r in items if r['device']==device and r['source_index']==i]
    config=lambda r: 'ort_p'+str(r['max_parallel_nodes']) if fid=='ort_genai_cpu_phi35' else meta[r['config']]['configuration']
    assert len(group)==len(expected_configs[fid]) and {config(r) for r in group}==expected_configs[fid]
    good=[r for r in group if r['success']]
    winner=min(good,key=lambda r:(r['wall_ms'],r['config'])) if good else None
    for r in group:
     cells.append(dict(device=device,configuration=config(r),parallel=r['max_parallel_nodes'],success=r['success'],
      complete_wall_ms=r['wall_ms'] if r['success'] else None,attempt_wall_ms=r['wall_ms'],
      failure_reason=r.get('failure_reason',''),result_json=r['result_json'],sha256=hashes[r['result_json']]))
    if winner:
     assert winner['state_matches_reference'] and winner['tool_outputs_match_reference'] and winner['generation_valid']
     cell=dict(framework_id=fid,device=device,source_index=i,configuration=config(winner),
      parallel=winner['max_parallel_nodes'],wall_ms=winner['wall_ms'],result_json=winner['result_json'],
      sha256=hashes[winner['result_json']],candidate_count=len(group),successful_candidates=len(good))
     winners.append(cell);selected.append(cell)
   successes=len(winners)
   point.update(framework_label=characterization.display_framework(fid,None),
    avg_wall_s=mean(w['wall_ms']/1000 for w in winners) if successes==len(DEVICES) else None,
    successes=successes,expected=len(DEVICES),sample_count=successes,device_count=len(DEVICES),model_count=1,combo_count=1,
    marker='' if successes==len(DEVICES) else f'ERR {successes}/{len(DEVICES)}',
    selected_cells=winners,candidate_cells=cells)
   point.pop('matched_cells',None)
 char['framework_offline_selection']={
  'run':str(run.relative_to(ROOT)),'ort_corrected_run':str(ort_run.relative_to(ROOT)),
  'sources':{str(p.relative_to(ROOT)):source(p) for p in [run/'manifest.json',run/'summary.json',run/'final_validation.json',
   run/'transport_provenance.json',ort_run/'summary.json',ort_run/'final_validation.json']},
  'workflow_sha256':manifest['workflow_sha256'],'workflow_indices':indices,
  'candidate_configurations_by_framework':{fid:sorted(configs) for fid,configs in expected_configs.items()},
  'selection':'For each framework/device/workflow select the fastest successful measured candidate; each bar equally averages both phone selections. No averaging across parallelism settings.',
  'scope':'llama.cpp uses six measured configurations at p=1..4; Genie and corrected ORT use measured p=1/2. Different candidate/backend/package spaces remain explicit.',
  'candidate_records':candidate_total,'candidate_successes':candidate_successes,'candidate_failures':candidate_total-candidate_successes,
  'expected_selected_records':len(DEVICES)*len(indices)*len(cohorts),'selected_records':len(selected),'selection_rows':selected,
  'original_failures_retained':True,'original_ORT_ERR_retained':correction['original_40_ERR_retained'],
  'new_experiments':0,'previous_matched_average_by_workflow':original_frames,
  'transport':'Original USB/wireless provenance retained; selected times include host/ADB overhead.'}
 return result

def apply_model_characterization(data,run:Path):
 """Restore all eight historical packages using the complete audited cohort."""
 manifest=read(run/'manifest.json');audit=read(run/'final_validation.json')
 analysis=read(run/'analysis/summary.json');assert audit['passed']
 assert set(manifest['catalog']['fig16_models'])==set(MODEL_CHARACTERIZATION_MODELS)
 rows=read(run/'summary.json')['rows']+[item['row'] for item in manifest['reused']]
 meta={job['case']['id']:job['case'] for job in manifest['all_jobs']}
 candidates=[r for r in rows if r['model_id'] in MODEL_CHARACTERIZATION_MODELS and
  meta[r['config']]['configuration'] in CANDIDATES and not meta[r['config']].get('repeat')]
 by={(r['device'],r['model_id'],r['source_index'],meta[r['config']]['configuration']):r for r in candidates}
 indices=manifest['catalog']['workflow_indices']
 assert len(candidates)==len(by)==800
 expected={(d,m,i,c) for d in DEVICES for m in MODEL_CHARACTERIZATION_MODELS for i in indices for c in CANDIDATES}
 assert set(by)==expected
 for r in candidates:assert digest(r['result_json'])==audit['all_original_sha256'][r['result_json']]
 selected={(s['device'],s['model_id'],s['source_index']):s for s in analysis['odabench_selection']}
 source_rows=[];selections=[];llm_count=0
 for d in DEVICES:
  for model in MODEL_CHARACTERIZATION_MODELS:
   for pos,i in enumerate(indices):
    s=selected[d,model,i];group=[by[d,model,i,c] for c in CANDIDATES]
    good=[r for r in group if r['success']];winner=s['winner']
    assert len(group)==5 and good and s['candidates']==5
    assert winner in good and winner['wall_ms']==min(r['wall_ms'] for r in good)
    raw=read(winner['result_json']);rr=raw['summary']
    assert rr['success'] and rr['state_matches_reference'] and rr['tool_outputs_match_reference']
    for step in raw['execution']['steps']:
     if step['kind']!='llm':continue
     llm_count+=1;q=step['raw'];output=q.get('stdout','');errors=output+'\n'+q.get('stderr','')
     assert step['success'] and q['returncode']==0 and q['response_text'].strip()
     assert not re.search(r'failed to initialize samplers|failed to load model|grammar sampler error|error loading model',errors,re.I)
     rate=re.search(r'Generation:\s*([0-9.]+)\s*t/s',output);assert rate and float(rate.group(1))>0
    source_rows.append(dict(device=d,model_id=model,framework_id='llama_cpp_selected',wall_s=winner['wall_ms']/1000,
     workflow_index=pos,workflow_id=winner['workflow_id'],source_index=i,selected_configuration=meta[winner['config']]['configuration']))
    selections.append(dict(device=d,model_id=model,source_index=i,configuration=meta[winner['config']]['configuration'],
     p=winner['max_parallel_nodes'],wall_ms=winner['wall_ms'],result_json=winner['result_json'],
     candidate_count=5,successful_candidates=len(good)))
 result=copy.deepcopy(data);char=result['characterization']
 char.update(model_order=MODEL_CHARACTERIZATION_MODELS,model_source_rows=source_rows,
  model_by_workflow=characterization.build_model_by_workflow(source_rows,MODEL_CHARACTERIZATION_MODELS))
 assert len(source_rows)==160 and len(char['model_by_workflow'])==10
 for frame in char['model_by_workflow']:
  assert len(frame['values'])==8 and all(v['sample_count']==v['device_count']==2 for v in frame['values'])
 char['datasets_by_figure']['model']=str(run.relative_to(ROOT))
 char['model_characterization_completion']={
  'run':str(run.relative_to(ROOT)),'historical_model_reference_commit':MODEL_REFERENCE_COMMIT,
  'historical_reference_scope':'Restore the July model package list/order only; July latency values and workflow subset are not reused.',
  'sources':{name:source(run/name) for name in ['manifest.json','summary.json','analysis/summary.json','final_validation.json','transport_provenance.json']},
  'models':MODEL_CHARACTERIZATION_MODELS,'candidate_configurations':CANDIDATES,'candidate_records':800,
  'new_records':sum(not r.get('reused',False) for r in candidates),'reused_records':sum(bool(r.get('reused')) for r in candidates),
  'successes':sum(r['success'] for r in candidates),'failures':sum(not r['success'] for r in candidates),
  'missing_conditions':0,'duplicate_conditions':0,'new_inference_calls_for_this_update':0,
  'observations_per_candidate_condition':1,'selected_records':160,'successful_selected_llm_nodes':llm_count,
  'selection':'For each device/model/workflow choose the fastest successful result among the same five configurations; retain all failed attempts.',
  'aggregation':'Each plotted model/workflow averages its two successful phone selections equally (2/2), not five candidates or repeated trials.',
  'transport':'Original USB/wireless provenance is retained; host wall time includes ADB overhead. Not a fully identical transport environment.',
  'selection_rows':selections}
 return result

def render(data, preview_dir=None, only='all'):
 """Render the active evaluation panels from separately supplied frozen data."""
 if only in ['all','execution']:
  execution.plot_execution_latency(data['latency'],preview_dir)
  followup.draw_timeline(data['followup'],preview_dir)
 if only in ['all','characterization']:
  characterization.plot_hardware_by_workflow(data['characterization'],CH,preview_dir)
 if only in ['all','characterization','model']:
  characterization.plot_model_by_workflow(data['characterization'],CH,preview_dir)
 if only in ['all','characterization','framework']:
  characterization.plot_framework_by_workflow(data['characterization'],CH,preview_dir)

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--refresh',action='store_true');p.add_argument('--run',type=Path,default=DEFAULT_RUN);p.add_argument('--preview-dir',type=Path)
 p.add_argument('--ort-correction',type=Path,help='Audited single-attempt ORT correction cohort for Fig.15 only.')
 p.add_argument('--model-characterization',type=Path,help='Complete audited eight-package cohort for the Model Characterization figure only.')
 p.add_argument('--framework-selection',type=Path,help='Audited cohort for per-phone/workflow framework offline selection.')
 p.add_argument('--only',choices=['all','execution','characterization','followup','framework','model'],default='all');a=p.parse_args()
 saved=read(SNAPSHOT);data=build(a.run.resolve()) if a.refresh else saved
 correction=a.ort_correction
 if a.refresh and correction is None:
  previous=saved['characterization'].get('framework_ort_correction')
  if previous:correction=ROOT/previous['run']
 if correction:
  data=apply_ort_correction(data,correction.resolve())
  if a.refresh and saved['characterization'].get('framework_ort_correction'):
   data['characterization']['framework_ort_correction']['original_failed_cells']=saved['characterization']['framework_ort_correction']['original_failed_cells']
 model_run=a.model_characterization
 if a.refresh and model_run is None:
  previous=saved['characterization'].get('model_characterization_completion')
  if previous:model_run=ROOT/previous['run']
 if model_run:data=apply_model_characterization(data,model_run.resolve())
 framework_run=a.framework_selection
 previous=saved['characterization'].get('framework_offline_selection')
 if (a.refresh or correction) and framework_run is None and previous:framework_run=ROOT/previous['run']
 if framework_run:
  data=apply_framework_selection(data,framework_run.resolve())
  if previous:
   data['characterization']['framework_offline_selection']['previous_matched_average_by_workflow']=previous['previous_matched_average_by_workflow']
 if a.refresh or correction or model_run or framework_run:write(SNAPSHOT,data)
 render(data,a.preview_dir,a.only)
 name='Framework Characterization' if a.only=='framework' else 'Model Characterization' if a.only=='model' else 'Evaluation figures'
 print('Updated '+name+' from the frozen audited data source; original rendering retained.')
if __name__=='__main__':main()
