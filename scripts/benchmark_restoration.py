"""Paired stage ablations, mask metrics, optional DDColor comparison. No installs."""
import argparse
import csv
from dataclasses import asdict, replace
import importlib.util
import json
from statistics import median
from pathlib import Path
import sys
from time import perf_counter

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from demo.photo_restoration import RestorationOptions, restore_pipeline
from demo.restoration_fixtures import make_cases


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',default=str(ROOT/'reports/restoration/paired'))
    parser.add_argument('--side',type=int,default=256)
    parser.add_argument('--cases',nargs='*')
    parser.add_argument('--classic',action='store_true')
    parser.add_argument('--colorize',action='store_true')
    parser.add_argument('--device',default='cpu',choices=['cpu','cuda'])
    parser.add_argument('--color-repeats',type=int,default=1)
    args=parser.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    spec=importlib.util.spec_from_file_location('baseline',ROOT/'reports/restoration/baseline_v2.py')
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    color,clean,cases=make_cases(args.side)
    def save(path,image):
        path=out/path
        cv2.imwrite(str(path),cv2.cvtColor(image,cv2.COLOR_RGB2BGR) if image.ndim==3 else image)
        return str(path.resolve())
    reference=save('reference-gray.png',clean)
    color_reference=save('reference-color.png',color)
    rows=[];details={}
    def add(case,method,stage,image,target,target_path,seconds,precision=None,recall=None):
        path=save(f'{case}-{method}-{stage}.png',image)
        rows.append({'case':case,'method':method,'stage':stage,
                     'psnr':float(peak_signal_noise_ratio(target,image,data_range=255)),
                     'ssim':float(structural_similarity(target,image,data_range=255,channel_axis=-1,
                                gaussian_weights=True,sigma=1.5,use_sample_covariance=False)),
                     'lpips':None,'seconds':seconds,'mask_precision':precision,'mask_recall':recall,
                     'image':path,'reference':target_path})
    def persist():
        (out/'results.json').write_text(json.dumps({'rows':rows,'details':details},indent=2),encoding='utf-8')
        if rows:
            with (out/'results.csv').open('w',newline='',encoding='utf-8') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    options=RestorationOptions()
    if args.classic:
        options=replace(options,detector='opencv',repair_backend='telea')
    colorizer=None
    if args.colorize:
        import torch
        from ddcolor import DDColor,ColorizationPipeline,build_ddcolor_model
        torch.set_num_threads(2)
        model=build_ddcolor_model(DDColor,model_path=str(ROOT/'modelscope/damo/cv_ddcolor_image-colorization/pytorch_model.pt'),
                                  input_size=512,model_size='large',device=torch.device(args.device))
        colorizer=ColorizationPipeline(model,input_size=512)
        if args.device=='cuda':
            torch.cuda.reset_peak_memory_stats()
        colorizer.process(cv2.cvtColor(clean,cv2.COLOR_RGB2BGR))  # warm-up excluded
    for name,(damaged,known_mask) in cases.items():
        if args.cases and name not in args.cases:
            continue
        print('Case:',name,flush=True)
        save(f'{name}-known-mask.png',known_mask)
        add(name,'input','source',damaged,clean,reference,0)
        start=perf_counter();old_result,old_mask=old.restore_photo(damaged);old_seconds=perf_counter()-start
        add(name,'old','final',old_result,clean,reference,old_seconds)
        restored=restore_pipeline(damaged,options)
        save(f'{name}-detected-mask.png',restored.mask)
        gt=known_mask>0;pred=restored.mask>0
        tp=int(np.count_nonzero(gt&pred))
        precision=tp/max(1,int(np.count_nonzero(pred)))
        recall=tp/max(1,int(np.count_nonzero(gt))) if np.any(gt) else None
        for stage in ('Repaired','Denoised','Contrast','Sharpened'):
            timing_key={'Repaired':'Repair','Denoised':'Denoise','Sharpened':'Sharpen'}.get(stage,stage)
            add(name,'new',stage,restored.stages[stage],clean,reference,restored.timings[timing_key],precision,recall)
        add(name,'new','final',restored.image,clean,reference,restored.timings['Total'],precision,recall)
        # Isolate sharpening from detection/repair/denoise, including when off by default.
        sharp=restore_pipeline(restored.stages['Contrast'],replace(options,detect=False,repair=False,
                               denoise=False,contrast=False,sharpen=True))
        add(name,'sharpen-ablation','final',sharp.image,clean,reference,sharp.timings['Total'])
        details[name]={'options':asdict(options),'timings':restored.timings,'notes':restored.notes,
                       'false_positive_fraction':float(np.mean(pred & ~gt))}
        if np.any(gt):
            for backend in ('telea','lama') if not args.classic else ('telea',):
                oracle=restore_pipeline(damaged,replace(options,repair_backend=backend,denoise=False,
                                          contrast=False,sharpen=False,mask_grow=0),manual_mask=known_mask)
                add(name,'oracle-'+backend,'repair',oracle.image,clean,reference,oracle.timings['Total'])
        if colorizer:
            for method,source,restore_seconds in [('DDColor',damaged,0),('Restoration-DDColor',restored.image,restored.timings['Total'])]:
                measurements=[]
                for _ in range(max(1,args.color_repeats)):
                    start=perf_counter();bgr=colorizer.process(cv2.cvtColor(source,cv2.COLOR_RGB2BGR))
                    measurements.append(perf_counter()-start)
                seconds=median(measurements)
                add(name,method,'color',cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB),color,color_reference,seconds+restore_seconds)
                details[name][method+'_color_timings']=measurements
            details[name]['color_device']=args.device
            if args.device=='cuda':
                details[name]['peak_cuda_allocated_mb']=torch.cuda.max_memory_allocated()/1024**2
                details[name]['peak_cuda_reserved_mb']=torch.cuda.max_memory_reserved()/1024**2
        persist()
    print('Saved',len(rows),'measurements to',str(out),flush=True)


if __name__=='__main__':
    main()
