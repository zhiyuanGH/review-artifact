"""Paper coverage matrix. Pure local planning; no device access or inference."""
import copy
import json
import os
from pathlib import Path

DEVICE_B = os.environ.get('MOBENCH_DEVICE_B_SERIAL','DEVICE_B_SERIAL')
CATALOG = Path(__file__).resolve().parents[1] / 'configs/selected10_paper_experiments.json'
CONFIGS = {'naive': ('rr', 1), 'wave2': ('wave', 2), 'wave3': ('wave', 3),
           'max_rr': ('rr', 4), 'max_wave': ('wave', 4), 'wave1': ('wave', 1)}
SUITES = ('models', 'hardware', 'framework', 'placement', 'npu', 'structure', 'resources', 'all')
QWEN = 'qwen1_5_1_8b_chat_q4_0'
PHI = 'phi_3_5_mini_instruct_q4_k_m'


def load_catalog(path=CATALOG):
    return json.loads(Path(path).read_text())


def llama_case(model, config, catalog, **extra):
    placement, parallel = CONFIGS[config] if config in CONFIGS else config.rsplit('_p', 1)
    return dict(id=model+'__'+config, model_id=model, runtime='llama_cli',
                parallel=int(parallel), placement=placement, configuration=config,
                model_path=catalog['models'][model]['model_path'], **extra)


def cases(suite, catalog=None, resource_configuration='max_wave'):
    catalog = catalog or load_catalog()
    if suite not in SUITES:
        raise ValueError('Unknown suite: '+suite)
    result = {}

    def add(case, figure):
        identity = case['id']
        if identity not in result:
            result[identity] = dict(case, figures=[])
        if figure not in result[identity]['figures']:
            result[identity]['figures'].append(figure)

    for name, key, figure in [('models','fig16_models','Fig.16'), ('hardware','fig14_models','Fig.14')]:
        if suite in (name, 'all'):
            for model in catalog[key]:
                for config in ('naive','wave2','wave3','max_rr','max_wave'):
                    add(llama_case(model, config, catalog), figure)
    if suite in ('framework','all'):
        for p in (1,2):
            add(llama_case(PHI, f'wave{p}', catalog), 'Fig.15')
            add(dict(id=f'phi35_ort_p{p}', model_id='phi35_mini_ort_cpu_int4',
                     runtime='ort_genai_cli', parallel=p, placement='cpu', configuration=f'ort_p{p}',
                     model_path='/data/local/tmp/ort_phi35_cpu_int4',
                     runner_dir='/data/local/tmp/oga_generate', jar='oga-generate.jar'), 'Fig.15')
            add(dict(id=f'phi35_genie_p{p}', model_id='phi35_mini_genie_npu',
                     runtime='genie_cli', parallel=p, placement='npu', configuration=f'genie_p{p}',
                     bundle_dir='/data/local/tmp/genie_bundle', config='auto', backend='npu'), 'Fig.15')
    if suite in ('placement','npu','all'):
        for unit in ('cpu','gpu','npu') if suite != 'npu' else ('npu',):
            for p in (1,3) if suite != 'npu' else (3,):
                case = llama_case(QWEN, f'{unit}_p{p}', catalog)
                add(case, 'Fig.6' if p == 3 else 'standalone-backend control')
                if unit == 'npu' and p == 3:
                    add(case, 'Fig.12')
        # Same p=3 reference isolates placement; it already exists in the main run.
        add(llama_case(QWEN, 'wave3', catalog), 'Fig.12')
        if suite != 'npu':
            add(llama_case(QWEN, 'wave3', catalog), 'Fig.6')
    if suite in ('structure','all'):
        for model in catalog['execution_models']:
            for config in ('wave1','wave2','wave3','max_wave'):
                add(llama_case(model, config, catalog), 'Fig.11')
    if suite in ('resources','all'):
        # A new repeated cohort; old one-off measurements are not reused as repeats.
        for config, figure, role in [('cpu_p3','Fig.6','cpu_p3'), ('gpu_p3','Fig.6','gpu_p3'),
                                     ('wave3','Fig.6','wave3'), ('naive','Fig.13','naive'),
                                     (resource_configuration,'Fig.13','odabench')]:
            for repeat in (1,2,3):
                case = llama_case(QWEN, config, catalog, indices=[179], devices=[DEVICE_B],
                                  repeat=repeat, role=role)
                case['id'] += f'__{figure.replace(".", "")}_{case["role"]}_repeat{repeat}'
                add(case, figure)
    return list(result.values())


def comparable_config(cfg):
    """Ignore only transport locations, retain numerical and placement parameters."""
    result = copy.deepcopy(cfg)
    for c in [result['llm']['llama_cli'], *result['llm']['llama_cli_overrides'].values()]:
        for key in ('adb','remote_prompt_dir','timeout'):
            c.pop(key, None)
        c.setdefault('hexagon_env', False)
    for c in result['qnn_tools'].values():
        c['command'] = ['native-phone-tool', '--state', 'fresh-initial-state']
    return result
