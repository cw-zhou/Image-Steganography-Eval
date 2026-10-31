"""Five new ECC images; reuse frozen baseline and randomized-header controls."""
import argparse
import hashlib
import io
import json
import random
import traceback
import uuid
from contextlib import redirect_stdout
from datetime import datetime,timezone
from pathlib import Path
from PIL import Image
from core.steganography import SteganographyCore
from ecc_payload_candidate import ECCPayloadCore
from run_comparison import aggregate,write_csv
from run_robustness_pilot import run_case,write_json,accuracy

METHODS=('baseline','header_randomized','payload_ecc')


def load_controls(header_root,baseline_root,cases,project):
    if not 1<=cases<=5:
        raise ValueError('Only 1..5 cases supported')
    hm=json.loads((header_root/'manifest.json').read_text(encoding='utf-8'))
    bm=json.loads((baseline_root/'manifest.json').read_text(encoding='utf-8'))
    if hm.get('seed')!=74000 or bm.get('seed')!=74000 or hm.get('cases')!=5 or bm.get('cases')!=5:
        raise ValueError('Both controls must be first-batch five-case runs, seed=74000')
    if hm['prompts']!=bm['prompts']:
        raise ValueError('Control prompts differ')
    hp=hm['provenance']
    for key in ('model_files','packages','attacks'):
        if hp[key]!=bm[key]:
            raise ValueError('Controls differ: '+key)
    # All previously recorded common source files must still be identical.
    for record in (hp['source_sha256'],bm['source_sha256']):
        for name,digest in record.items():
            if hashlib.sha256((project/name).read_bytes()).hexdigest()!=digest:
                raise ValueError('Source changed since controls: '+name)
    hr=json.loads((header_root/'results.json').read_text(encoding='utf-8'))
    br=json.loads((baseline_root/'results.json').read_text(encoding='utf-8'))
    rows=[]
    for i in range(cases):
        seed=74000+i
        for method,source in [('baseline',br),('header_randomized',hr)]:
            found=[r for r in source if r['method']==method and r['seed']==seed]
            if len(found)!=1 or found[0]['generation_error'] is not None or found[0]['niqe'] is None:
                raise ValueError('Incomplete source control')
            row=dict(found[0])
            if row['prompt']!=hm['prompts'][i]:
                raise ValueError('Result prompt mismatch')
            row.update(reused_control=True,source_run=str(baseline_root if method=='baseline' else header_root))
            rows.append(row)
    return hm,rows


def run_new(plan,controls,core,attack_fn,niqe_fn,scorer,out,seed_fn):
    rows=list(controls)
    write_json(out/'results.json',rows)
    for i,prompt in enumerate(plan['prompts']):
        seed=74000+i
        rng=random.Random(seed)
        bits=''.join(str(rng.randrange(2)) for _ in range(8192))
        control=next(r for r in controls if r['method']=='header_randomized' and r['seed']==seed)
        if hashlib.sha256(bits.encode('ascii')).hexdigest()!=control['payload_sha256']:
            raise ValueError('Message differs from frozen header control')
        directory=out/('case_{:02d}_payload_ecc'.format(i))
        core.extraction_diagnostics=[]
        if callable(getattr(core._backend,'prepare',None)):
            core._backend.prepare(directory)
        print('\nCASE {}/{} / payload_ecc'.format(i+1,len(plan['prompts'])),flush=True)
        row=run_case(core,attack_fn,directory,prompt,bits,3,seed,seed_fn,header_bits=1984)
        row.update(method='payload_ecc',case_index=i,payload_sha256=control['payload_sha256'],
                   niqe=None,niqe_error=None,reused_control=False)
        if row['generation_error'] is None:
            context=json.loads((directory/'context.json').read_text(encoding='utf-8'))
            original=json.loads((Path(control['source_run'])/('case_{:02d}_header_randomized'.format(i))/'context.json').read_text(encoding='utf-8'))
            compared=dict(context)
            compared.pop('payload_code')
            if compared!=original:
                raise ValueError('ECC context differs beyond payload code identity')
            diagnostics=[]
            for attack,d in zip(('clean','jpeg','awgn','mblur'),core.extraction_diagnostics):
                d=dict(d)
                raw=d.pop('raw_recovered_bits',None)
                if raw is not None:
                    (directory/(attack+'_raw_recovered.txt')).write_text(raw,encoding='ascii')
                    d['raw_acc_same_ecc_image']=accuracy(bits,raw)
                    d['ecc_acc_same_ecc_image']=row['attacks'][attack]['acc']
                d['attack']=attack
                diagnostics.append(d)
            row['payload_decoder_diagnostics']=diagnostics
            try:
                with Image.open(directory/'stego.png') as image:
                    row['niqe']=float(niqe_fn(image.convert('RGB')))
            except Exception as exc:
                row['niqe_error']=str(exc)
                (directory/'niqe_traceback.txt').write_text(traceback.format_exc(),encoding='utf-8')
        write_json(directory/'result.json',row)
        rows.append(row)
        write_json(out/'results.json',rows)
        write_csv(out/'results.csv',rows)
        write_json(out/'status.json',dict(status='running',completed_new_cases=i+1))
    with redirect_stdout(io.StringIO()) as log:
        summary=aggregate(rows,METHODS,scorer)
    (out/'scorer_output.txt').write_text(log.getvalue(),encoding='utf-8')
    write_json(out/'summary.json',dict(status='complete',methods=summary,
        note='Same first-batch development prompts/seeds. Baseline carries 16384 bits; others 8192. Old controls reused only after provenance checks.'))
    for method,item in summary.items():
        print('\n{}: bpp={}, NIQE={}, score={}'.format(method,item['mean_metrics']['bpp'],item['mean_metrics']['niqe'],item['provisional_score']))
        for a,value in item['mean_acc_including_failures'].items():
            print('  {}: Acc={:.4f}, failures={}/{}'.format(a,value,item['extraction_failures'][a],item['cases']))
    failed=any(not x['niqe_complete'] or any(x['extraction_failures'].values()) for x in summary.values())
    write_json(out/'status.json',dict(status='complete_with_failures' if failed else 'complete',completed_new_cases=len(plan['prompts'])))
    return 1 if failed else 0


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--header-run',type=Path,default=Path('outputs/header-followup10-a/run_20261031T103156Z_46bcfd14'))
    parser.add_argument('--baseline-run',type=Path,default=Path('outputs/baseline-followup10-a/run_20261031T104021Z_86b817f9'))
    parser.add_argument('--cases',type=int,default=5)
    parser.add_argument('--plan-only',action='store_true')
    parser.add_argument('--output-dir',type=Path,default=Path('outputs/payload-ecc'))
    args=parser.parse_args(argv)
    project=Path(__file__).resolve().parent
    header_root=args.header_run.resolve();baseline_root=args.baseline_run.resolve()
    hm,controls=load_controls(header_root,baseline_root,args.cases,project)
    plan=dict(prompts=hm['prompts'][:args.cases],seed=74000,cases=args.cases,
              methods=list(METHODS),new_generation_count=args.cases,new_inversion_count=args.cases*4,
              reused_control_cases=len(controls),payload_bits=8192,parity_bits=6208,header_bits=1984,
              source_runs=dict(header=str(header_root),baseline=str(baseline_root)),
              source_manifest_sha256={str(r):hashlib.sha256((r/'manifest.json').read_bytes()).hexdigest() for r in (header_root,baseline_root)},
              new_source_sha256={n:hashlib.sha256((project/n).read_bytes()).hexdigest() for n in ('sparse_payload_code.py','ecc_payload_candidate.py','run_payload_ecc_pilot.py')},
              hypothesis='Does actual payload parity improve total course score versus original baseline? This is a development test, not final held-out evaluation.')
    if args.plan_only:
        print(json.dumps(plan,ensure_ascii=False,indent=2));return 0
    import importlib.metadata
    import torch
    from run_robustness_pilot import seed_everything
    from evaluation.attacks import apply_attack
    from evaluation.metrics import calculate_niqe
    from evaluation.scorer import calculate_final_score
    from run_header_ablation import RecordingBackend
    if not torch.cuda.is_available():raise RuntimeError('CUDA GPU required')
    for name,version in hm['provenance']['packages'].items():
        if importlib.metadata.version(name)!=version:raise ValueError('Package changed: '+name)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    paths=hm['provenance']['model_files']
    base=SteganographyCore(paths['ckpt'],paths['clip'],paths['config'],device='cuda')
    core=ECCPayloadCore.from_backend(base._backend,device=base.device)
    core._backend=RecordingBackend(base._backend)
    out=args.output_dir/('run_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8])
    out.mkdir(parents=True,exist_ok=False)
    write_json(out/'manifest.json',dict(plan,source_provenance=hm['provenance']))
    print('OUTPUT:',out.resolve(),flush=True)
    try:
        exit_code=run_new(plan,controls,core,apply_attack,calculate_niqe,calculate_final_score,out,seed_everything)
    except Exception:
        (out/'traceback.txt').write_text(traceback.format_exc(),encoding='utf-8')
        write_json(out/'status.json',dict(status='error'));raise
    print('Report:',(out/'summary.json').resolve());return exit_code


if __name__=='__main__':raise SystemExit(main())
