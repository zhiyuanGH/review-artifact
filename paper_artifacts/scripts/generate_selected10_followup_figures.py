"""Refresh follow-up paper figures from the same selected-ten measurements.

No device calls. --refresh audits raw records and freezes derived inputs; the
default renders that snapshot. Missing NPU-only and Phi-3.5 measurements are
never replaced by round-robin data or relabeled historical workflows.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import figure_style
import generate_studies_characterization_figures as char
import generate_studies_execution_figures as execution

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "local_data/selected10_four_methods_20260928"
OUT = ROOT / "figures/studies/execution"
SNAPSHOT = OUT / "selected10_followup_provenance.json"
DEVICES = ["elite", "gen3"]
MODELS = ["qwen3_0_6b_q4_k_m", "llama_3_2_1b_instruct_q4_0", "qwen1_5_1_8b_chat_q4_0"]
GPU_SAMPLE_POLICY = (
    "Recorded finite GPU percentages are preserved. Blank percentages with "
    "both GPU counters equal to zero and no sample error are shown as 0% "
    "(zero-counter idle convention); other unavailable readings remain gaps. "
    "Sample averages include these zero-counter intervals."
)


def read(path):
    return json.loads(path.read_text())


def write(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False)+"\n")


def parse_gpu_busy_percent(record):
    """Distinguish the collector's zero-counter intervals from missing reads."""
    def finite_number(value):
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if math.isfinite(number) else None

    raw_percentage = record.get('gpu_busy_percent')
    recorded = finite_number(raw_percentage)
    if recorded is not None:
        return recorded
    if (not (raw_percentage or '').strip()
            and not (record.get('sample_error') or '').strip()
            and finite_number(record.get('gpu_busy_counter')) == 0
            and finite_number(record.get('gpu_total_counter')) == 0):
        return 0.0
    return None


def read_resource_samples(path, start, end):
    samples = []
    zero_counter_count = 0
    with Path(path).open(newline='') as handle:
        for record in csv.DictReader(handle):
            timestamp = float(record['timestamp_epoch_s'])
            if not start <= timestamp <= end:
                continue
            point = {'elapsed_s': timestamp-start}
            for key, source, divisor in [('cpu_pct','cpu_util_percent',1), ('gpu_pct','gpu_busy_percent',1), ('mem_gb','mem_used_mb',1024)]:
                if key == 'gpu_pct':
                    point[key] = parse_gpu_busy_percent(record)
                else:
                    value = float(record[source])/divisor if record[source] else None
                    point[key] = value if value is not None and math.isfinite(value) else None
            if not (record.get('gpu_busy_percent') or '').strip() and point['gpu_pct'] == 0:
                zero_counter_count += 1
            samples.append(point)
    samples.sort(key=lambda point: point['elapsed_s'])
    return samples, zero_counter_count


def build():
    latency = read(OUT / "execution_latency_selected10.json")
    indices = [r["source_index"] for r in latency["rows"]]
    selected = latency["latency"]["selection"]
    sources = dict(latency["latency"]["sources"])
    raw = []
    for d in DEVICES:
        path = RUN / d / "summary.json"
        if hashlib.sha256(path.read_bytes()).hexdigest() != sources[d]["sha256"]:
            raise ValueError("Raw source differs from latency figure snapshot")
        summary = read(path)
        assert summary["complete"] and len(summary["rows"]) == 150
        raw += summary["rows"]
    keys = {(r['device'], r['model_id'], r['source_index'], r['config']) for r in raw}
    assert len(keys) == len(raw) == 300
    chosen = {(s['device'], s['model_id'], s['source_index']): s for s in selected}
    assert len(chosen) == len(selected) == 60
    source_rows = []
    for s in selected:
        options = [r for r in raw if (r['device'], r['model_id'], r['source_index']) ==
                   (s['device'], s['model_id'], s['source_index']) and r['success']]
        assert s['wall_ms'] == min(r['wall_ms'] for r in options)
        assert any(r['config'] == s['configuration'] and r['wall_ms'] == s['wall_ms'] for r in options)
        source_rows.append({"device": s['device'], "model_id": s['model_id'],
                            "framework_id": "llama_cpp_selected", "wall_s": s['wall_ms']/1000,
                            "workflow_index": indices.index(s['source_index']),
                            "workflow_id": s['workflow_id'], "source_index": s['source_index'],
                            "selected_configuration": s['configuration']})
    hardware = char.build_hardware_by_workflow(source_rows, DEVICES)
    models = char.build_model_by_workflow(source_rows, MODELS)
    structure = []
    for d in DEVICES:
        for m in [MODELS[1], MODELS[2], MODELS[0]]:
            ours = [s for s in selected if s['device'] == d and s['model_id'] == m]
            blind = [r for r in raw if r['device'] == d and r['model_id'] == m and r['config'] == 'wave3']
            assert len(ours) == len(blind) == 10
            passed = sum(r['success'] for r in blind)
            a = sum(s['wall_ms'] for s in ours)/60000
            b = sum(r['wall_ms'] for r in blind)/60000
            label = {MODELS[0]:'Qwen3', MODELS[1]:'Llama', MODELS[2]:'Qwen1.5'}[m]
            structure.append({'device': d, 'model_id': m,
                              'label': label+'\n'+('8E' if d==DEVICES[0] else '8G3'),
                              'odabench_total_wall_min': a, 'blind_p3_attempt_wall_min': b,
                              'blind_p3_total_wall_min': b if passed == 10 else None,
                              'blind_p3_successes': passed, 'total': 10,
                              'blind_p3_clean': passed == 10,
                              'blind_p3_over_odabench': b/a if passed == 10 else None,
                              'odabench_parallel': 0})
    clean = [r for r in structure if r['blind_p3_clean']]
    a = sum(r['odabench_total_wall_min'] for r in clean)
    b = sum(r['blind_p3_total_wall_min'] for r in clean)
    summary = {'case_count': 6, 'clean_case_count': len(clean),
               'odabench_p2_case_count': 0, 'odabench_p3_case_count': 0,
               'odabench_total_min': a, 'blind_p3_total_min': b,
               'blind_p3_over_odabench': b/a,
               'latency_reduction_vs_blind_p3_pct': 100*(1-a/b),
               'max_case_over_odabench': max(r['blind_p3_over_odabench'] for r in clean),
               'comparison_scope': 'Totals only over phone/model suites with 10/10 successes for both methods.',
               'interpretation': 'Joint selected execution versus fixed wave p=3; not a placement-controlled structure-only ablation.'}
    # The longest selected workflow replaces historical A200. Keep one observed
    # run per method, unavailable readings, and whole-device memory semantics.
    device, model, index = DEVICES[1], MODELS[2], indices[-1]
    timeline = {'workflow_index': index, 'workflow_label': 'R179', 'device': device,
                'model_id': model, 'mode': 'single_observed_run', 'methods': {}, 'source_sha256': {},
                'gpu_sample_policy': GPU_SAMPLE_POLICY}
    for method, config in [('naive','naive'), ('odabench',chosen[(device,model,index)]['configuration'])]:
        folder = RUN/device/model/config/'resources'
        by = read(folder/'resource_by_workflow.json')[str(index)]
        start, end = by['started_epoch_s'], by['finished_epoch_s']
        row = next(r for r in raw if (r['device'],r['model_id'],r['source_index'],r['config']) == (device,model,index,config))
        assert row['success'] and by['status'] == 'completed'
        samples, zero_counter_count = read_resource_samples(folder/'resource_samples.csv', start, end)
        assert samples and all(any(s[k] is not None for s in samples) for k in ('cpu_pct','gpu_pct','mem_gb'))
        timeline['methods'][method] = {'configuration': config, 'duration_s': row['wall_ms']/1000,
                                      'sample_count': len(samples), 'samples': samples,
                                      'gpu_zero_counter_sample_count': zero_counter_count,
                                      'gpu_missing_sample_count': sum(s['gpu_pct'] is None for s in samples),
                                      'runs': [{'samples': samples}],
                                      'averages': {k:mean(s[k] for s in samples if s[k] is not None) for k in ('cpu_pct','gpu_pct','mem_gb')}}
        for name in ('resource_samples.csv','resource_by_workflow.json'):
            path = folder/name
            timeline['source_sha256'][str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {'dataset': RUN.name, 'source_indices': indices, 'source_sha256': sources,
            'selection': selected, 'hardware_by_workflow': hardware, 'model_by_workflow': models,
            'source_rows': source_rows, 'structure_ablation': {'rows': structure, 'summary': summary},
            'timeline': timeline, 'scope': 'Same 60 successful MOBench selections as the main latency figure.',
            'pending_measurements': ['Qwen1.5 NPU-only p=3', 'Phi-3.5 Genie/ORT/llama.cpp framework matrix']}


def draw_structure(data, preview):
    execution.configure_style()
    rows = data['structure_ablation']['rows']
    fig, ax = plt.subplots(figsize=(3.45,1.85))
    fig.subplots_adjust(left=.17,right=.99,bottom=.29,top=.79)
    for key, label, offset in [('odabench','MOBench',-.16),('blind_p3','fixed p=3',.16)]:
        values = [r[key+'_total_wall_min'] for r in rows]
        ax.bar([i+offset for i in range(6)], [v if v is not None else float('nan') for v in values], width=.32,
               color=figure_style.bar_color(key), edgecolor=figure_style.BAR_EDGE, linewidth=.35, label=label)
    ymax=max(r[k] for r in rows for k in ('odabench_total_wall_min','blind_p3_total_wall_min') if r[k] is not None)*1.16
    for i,r in enumerate(rows):
        if not r['blind_p3_clean']:
            ax.plot(i+.16,ymax*.025,'x',color='#B83232',ms=4)
            ax.text(i+.16,ymax*.15,f"ERR {r['blind_p3_successes']}/10",rotation=90,ha='center',va='bottom',fontsize=6,color='#B83232')
    ax.set_xticks(range(6),[r['label'] for r in rows])
    ax.set(xlabel='Device-model case',ylabel='Total latency (min)',ylim=(0,ymax),xlim=(-.55,5.45))
    execution.study_box_axes(ax)
    figure_style.boxed_legend_above_axes(fig,*ax.get_legend_handles_labels(),ncol=2,prop={'size':7})
    execution.save_figure(fig,OUT/'execution_structure_ablation',preview,exact_canvas=True,top_margin_points=2.0)


def draw_timeline(data, preview):
    execution.configure_style()
    timeline=data['timeline']
    completion_s=timeline['methods']['odabench']['duration_s']
    completion_color=figure_style.bar_color('odabench')
    fig,axes=plt.subplots(1,3,figsize=(5.35,1.48),sharex=True)
    fig.subplots_adjust(left=.09,right=.99,bottom=.28,top=.77,wspace=.48)
    for ax,(key,label) in zip(axes,[('cpu_pct','CPU (%)'),('gpu_pct','GPU (%)'),('mem_gb','Mem. (GB)')]):
        for method,name in [('naive','Sequential'),('odabench','MOBench')]:
            samples=timeline['methods'][method]['samples']
            ax.plot([s['elapsed_s'] for s in samples], [s[key] if s[key] is not None else float('nan') for s in samples],
                    color=figure_style.bar_color(method),lw=1,marker='o',ms=1.7,label=name)
        ax.set_ylabel(label,labelpad=1.4)
        execution.study_box_axes(ax)
        if key!='mem_gb': ax.set(ylim=(-3,103),yticks=[0,50,100])
        ax.axvline(completion_s,color=completion_color,lw=.85,ls='--',zorder=4)
    max_elapsed=max(m['duration_s'] for m in timeline['methods'].values())*1.02
    for ax in axes:
        ax.set(xlim=(0,max_elapsed),xticks=[0,60,120])
    fig.text(.54,.075,'Elapsed time (s)',ha='center',va='center',fontsize=execution.STUDY_FONT['label'])
    handles,labels=axes[0].get_legend_handles_labels()
    handles.append(plt.Line2D([],[],color=completion_color,lw=.85,ls='--'))
    labels.append(f'MOBench ends at {completion_s:.1f} s')
    figure_style.boxed_legend_above_axes(fig,handles,labels,ncol=3,prop={'size':7})
    execution.save_figure(fig,OUT/'execution_resource_usage',preview,exact_canvas=True,top_margin_points=2.0)


def publish(data, preview):
    draw_structure(data,preview)
    draw_timeline(data,preview)
    target=char.DEFAULT_OUTPUT_DIR
    metrics=read(target/'characterization_comparison_provenance.json')
    metrics.update({'model_order':MODELS,'device_order':DEVICES,
                    'hardware_by_workflow':data['hardware_by_workflow'], 'model_by_workflow':data['model_by_workflow'],
                    'model_source_rows':data['source_rows']})
    metrics['sources']['hardware_and_model_selected10']='figures/studies/execution/selected10_followup_provenance.json'
    metrics['datasets_by_figure']={'hardware':data['dataset'],'model':data['dataset'],'framework':'historical ten-workflow sample; replacement measurements pending'}
    char.plot_hardware_by_workflow(metrics,target,preview)
    char.plot_model_by_workflow(metrics,target,preview)
    write(target/'characterization_comparison_provenance.json',metrics)
    metrics=read(OUT/'execution_latency_provenance.json')
    metrics['structure_ablation']=data['structure_ablation']
    metrics['datasets_by_figure'].update({'structure_ablation':data['dataset'],'resource_usage':data['dataset'],
                                         'decider_ablation':'historical ten-workflow sample; replacement measurements pending'})
    write(OUT/'execution_latency_provenance.json',metrics)
    write(OUT/'execution_resource_usage_timeline_summary.json',data['timeline'])


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--refresh',action='store_true')
    ap.add_argument('--preview-dir',type=Path)
    args=ap.parse_args()
    from refresh_evaluation_sources import SNAPSHOT as evaluation_snapshot, DEFAULT_RUN, build as build_evaluation, render
    if evaluation_snapshot.exists():
        if args.refresh:
            write(evaluation_snapshot,build_evaluation(DEFAULT_RUN))
        render(read(evaluation_snapshot),args.preview_dir,only='followup')
        return
    if args.refresh:
        data=build();write(SNAPSHOT,data)
    else: data=read(SNAPSHOT)
    publish(data,args.preview_dir)
    print(json.dumps({'updated_figures':4,'source_indices':data['source_indices'],
                      'structure':data['structure_ablation']['summary'],
                      'timeline':{k:v['averages'] for k,v in data['timeline']['methods'].items()}},indent=2))


if __name__=='__main__': main()
