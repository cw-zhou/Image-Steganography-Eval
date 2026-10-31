"""Frozen 100-caption baseline/ECC comparison with per-method checkpoints."""
import argparse
import hashlib
import importlib.metadata
import io
import json
import math
import os
import random
import traceback
from contextlib import redirect_stdout
from datetime import datetime,timezone
from pathlib import Path
from PIL import Image
from core.steganography import SteganographyCore
from core.baseline_reference import BaselineReferenceCore
from ecc_payload_candidate import ECCPayloadCore
from run_comparison import aggregate,write_csv
from run_header_ablation import RecordingBackend
from run_robustness_pilot import run_case,seed_everything,accuracy

METHODS=('baseline','payload_ecc')
PROMPT_HASH='332419b39ada7bd659181c038df5d0a2282521da1f54f8c9670edda184f3b680'
CKPT_HASH='e1441589a6f3c5a53f5f54d0975a18a7feb7cdf0b0dee276dfc3331ae376a053'


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def atomic_json(path,data):
    temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    os.replace(str(temp),str(path))


def case_dir(out,index,method):
    return out/('case_{:03d}_{}'.format(index,method))


def commit_case(directory,row):
    atomic_json(directory/'result.json',row)
    hashes={p.name:sha(p) for p in directory.iterdir() if p.is_file() and p.name!='complete.json'}
    atomic_json(directory/'complete.json',dict(files=hashes))


def read_case(directory):
    marker=directory/'complete.json'
    if not marker.exists():return None
    for name,digest in json.loads(marker.read_text(encoding='utf-8'))['files'].items():
        p=directory/name
        if p.parent!=directory or not p.is_file() or sha(p)!=digest:
            raise ValueError('Completed case changed: '+str(p))
    return json.loads((directory/'result.json').read_text(encoding='utf-8'))


def collect(out):
    rows=[]
    for i in range(100):
        for method in METHODS:
            row=read_case(case_dir(out,i,method))
            if row is not None:
                if row['case_index']!=i or row['method']!=method:raise ValueError('Checkpoint identity mismatch')
                rows.append(row)
    return rows


def archive_partial(directory):
    if directory.exists():
        stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        directory.rename(directory.with_name('incomplete_'+directory.name+'_'+stamp))


def summarize(out,rows,scorer,status):
    complete=len(rows)==200
    with redirect_stdout(io.StringIO()) as log:
        summaries=aggregate(rows,METHODS,scorer)
    # Running averages are diagnostic; never label an incomplete run as final.
    if not complete:
        for item in summaries.values():item['provisional_score']=None
    atomic_json(out/'results.json',rows)
    write_csv(out/'results.csv',rows)
    atomic_json(out/'summary.json',dict(status=status,complete_100_pairs=complete,
        planned_method_cases=200,completed_method_cases=len(rows),methods=summaries,
        scope='Project-selected frozen COCO-caption evaluation. Different capacities: baseline16384, ECC8192. Not instructor-supplied data.'))
    atomic_json(out/'status.json',dict(status=status,completed_method_cases=len(rows),planned_method_cases=200))
    (out/'scorer_output.txt').write_text(log.getvalue() if complete else 'Final score withheld until all 100 pairs finish.\n',encoding='utf-8')
    return summaries


def load_plan(project):
    prompt_path=project/'data/prompts/final100_coco_frozen.txt'
    manifest_path=project/'data/prompts/final100_coco_manifest.json'
    source=json.loads(manifest_path.read_text(encoding='utf-8'))
    if sha(prompt_path)!=PROMPT_HASH or source['prompt_sha256']!=PROMPT_HASH:
        raise ValueError('Frozen prompt file changed')
    prompts=prompt_path.read_text(encoding='utf-8').splitlines()
    if len(prompts)!=100 or prompts!=[x['prompt'] for x in source['selected']]:
        raise ValueError('Frozen prompt order differs')
    if source['generation_seed_start']!=20261101:raise ValueError('Seed changed')
    for name,digest in source['frozen_candidate_sha256'].items():
        if sha(project/name)!=digest:raise ValueError('Candidate changed: '+name)
    files=['core/mapping.py','core/randomized_header.py','core/steganography.py','core/diffusion_backend.py',
           'core/baseline_reference.py','baseline/scripts/mapping_module.py','baseline/ldm/models/diffusion/ddpm.py',
           'baseline/configs/stable-diffusion/ldm.yaml','evaluation/attacks.py','evaluation/metrics.py',
           'evaluation/scorer.py','run_robustness_pilot.py','run_comparison.py','run_header_ablation.py',
           'sparse_payload_code.py','ecc_payload_candidate.py','run_final100.py']
    return dict(prompts=prompts,source=source,source_manifest_sha256=sha(manifest_path),methods=list(METHODS),
                seed_start=20261101,cases=100,new_generation_count=200,new_inversion_count=800,
                baseline_payload_bits=16384,candidate_payload_bits=8192,
                context=dict(ECCPayloadCore.DEFAULT_CONTEXT),
                source_sha256={n:sha(project/n) for n in files},
                packages={n:importlib.metadata.version(n) for n in ('torch','torchvision','numpy','scipy','Pillow','pyiqa')})


def run_jobs(plan,out,cores,attack_fn,niqe_fn,scorer,seed_fn,max_new_cases):
    rows=collect(out)
    summaries=summarize(out,rows,scorer,'running')
    new_pairs=0
    try:
        for i,prompt in enumerate(plan['prompts']):
            pending=[m for m in METHODS if read_case(case_dir(out,i,m)) is None]
            if not pending:continue
            if new_pairs>=max_new_cases:break
            seed=plan['seed_start']+i
            rng=random.Random(seed)
            full=''.join(str(rng.randrange(2)) for _ in range(16384))
            for method in pending:
                directory=case_dir(out,i,method)
                archive_partial(directory)
                core=cores[method]
                core._backend.prepare(directory)
                if method=='payload_ecc':core.extraction_diagnostics=[]
                bits=full if method=='baseline' else full[:8192]
                print('\nCASE {}/100 / {}'.format(i+1,method),flush=True)
                row=run_case(core,attack_fn,directory,prompt,bits,'baseline' if method=='baseline' else 3,
                             seed,seed_fn,header_bits=0 if method=='baseline' else 1984)
                row.update(method=method,case_index=i,payload_sha256=hashlib.sha256(bits.encode('ascii')).hexdigest(),niqe=None,niqe_error=None)
                if row['generation_error'] is None:
                    if method=='payload_ecc':
                        ds=[]
                        for attack,d in zip(('clean','jpeg','awgn','mblur'),core.extraction_diagnostics):
                            d=dict(d);raw=d.pop('raw_recovered_bits',None)
                            if raw is not None:
                                (directory/(attack+'_raw_recovered.txt')).write_text(raw,encoding='ascii')
                                d['raw_acc_same_ecc_image']=accuracy(bits,raw)
                                d['ecc_acc_same_ecc_image']=row['attacks'][attack]['acc']
                            d['attack']=attack;ds.append(d)
                        row['payload_decoder_diagnostics']=ds
                    try:
                        with Image.open(directory/'stego.png') as image:value=float(niqe_fn(image.convert('RGB')))
                        if not math.isfinite(value) or value<0:raise ValueError('Invalid NIQE')
                        row['niqe']=value
                    except Exception as exc:
                        row['niqe_error']=str(exc)
                        (directory/'niqe_traceback.txt').write_text(traceback.format_exc(),encoding='utf-8')
                commit_case(directory,row)
                rows.append(row)
                summarize(out,rows,scorer,'running')
            new_pairs+=1
        complete=len(rows)==200
        status='complete' if complete else 'checkpointed'
        summaries=summarize(out,rows,scorer,status)
        if complete and any(not x['niqe_complete'] or any(x['extraction_failures'].values()) for x in summaries.values()):
            summarize(out,rows,scorer,'complete_with_failures')
        print('\nCompleted method cases: {}/200'.format(len(rows)),flush=True)
        if complete:
            print('=== FINAL 100-PAIR SUMMARY ===')
            print(json.dumps(summaries,ensure_ascii=False,indent=2))
        else:print('Checkpoint saved. Resume the SAME run; do not create a new sample set.')
        print('Report:',(out/'summary.json').resolve())
        return summaries
    except BaseException:
        rows=collect(out)
        summarize(out,rows,scorer,'interrupted_or_error')
        (out/'interruption_traceback.txt').write_text(traceback.format_exc(),encoding='utf-8')
        raise


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan-only',action='store_true')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--max-new-cases',type=int,default=100)
    parser.add_argument('--run-dir',type=Path,default=Path('outputs/final100_coco_ecc'))
    args=parser.parse_args(argv)
    if not 1<=args.max_new_cases<=100:parser.error('--max-new-cases must be 1..100')
    project=Path(__file__).resolve().parent
    plan=load_plan(project)
    if args.plan_only:
        print(json.dumps({k:v for k,v in plan.items() if k not in ('source','prompts')},ensure_ascii=False,indent=2));return 0
    out=args.run_dir.resolve()
    if args.resume:
        saved=json.loads((out/'manifest.json').read_text(encoding='utf-8'))
        if saved['plan']!=plan:raise ValueError('Code, samples, context or packages changed since this run')
        collect(out)  # Verify completed artifacts before any model loading.
        if len(collect(out))==200:
            print('All 100 pairs already finished. Existing report:',out/'summary.json');return 0
    elif out.exists():raise FileExistsError('Run directory already exists; use --resume to continue')
    paths=dict(ckpt=project/'weights/v1-5-pruned.ckpt',clip=project/'weights/clip',config=project/'baseline/configs/stable-diffusion/ldm.yaml')
    if not paths['clip'].is_dir():raise FileNotFoundError(paths['clip'])
    print('Checking SD checkpoint SHA256; reading 7.7GB can take a little time...',flush=True)
    if sha(paths['ckpt'])!=CKPT_HASH:raise ValueError('SD checkpoint hash mismatch')
    model_files={n:dict(path=str(p),bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns) for n,p in paths.items()}
    # Include CLIP content fingerprints to reject changed model files on resume.
    clip_files={p.name:sha(p) for p in paths['clip'].iterdir() if p.is_file()}
    if args.resume and (saved['model_files']!=model_files or saved['clip_sha256']!=clip_files):
        raise ValueError('Model files changed since this run')
    import torch
    if not torch.cuda.is_available():raise RuntimeError('CUDA GPU required')
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    if not args.resume:
        out.mkdir(parents=True,exist_ok=False)
        atomic_json(out/'manifest.json',dict(plan=plan,model_files=model_files,clip_sha256=clip_files,
                    ckpt_sha256=CKPT_HASH,gpu=torch.cuda.get_device_name(0),created_utc=datetime.now(timezone.utc).isoformat()))
    from evaluation.attacks import apply_attack
    from evaluation.metrics import calculate_niqe
    from evaluation.scorer import calculate_final_score
    print('Loading model...',flush=True)
    base=SteganographyCore(paths['ckpt'],paths['clip'],paths['config'],device='cuda')
    baseline=BaselineReferenceCore(base._backend)
    ecc=ECCPayloadCore.from_backend(base._backend,device='cuda')
    baseline._backend=RecordingBackend(base._backend)
    ecc._backend=RecordingBackend(base._backend)
    run_jobs(plan,out,{'baseline':baseline,'payload_ecc':ecc},apply_attack,calculate_niqe,calculate_final_score,seed_everything,args.max_new_cases)
    return 0


if __name__=='__main__':raise SystemExit(main())
