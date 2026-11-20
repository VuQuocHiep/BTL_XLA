"""Real MobileSAM evidence; never substitute synthetic masks for inference."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
import torch
from skimage import data
from demo.object_recoloring import RecolorSession, recolor
from demo.object_segmentation import MobileSegmenter


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--ddcolor', action='store_true', help='Keep real DDColor on GPU and test routing')
    parser.add_argument('--device', default='cpu', choices=['cpu','cuda','auto'])
    args=parser.parse_args()
    torch.set_num_threads(4)
    folder=ROOT/'reports/recoloring'
    folder.mkdir(parents=True,exist_ok=True)
    base=ROOT/'third_party/MobileSAM'
    def sample(path):
        image=cv2.imread(str(base/path))
        if image is None: raise RuntimeError(path)
        return cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    cases=[('shirt',data.astronaut(),[(.60,.59),(.27,.65),(.48,.25),(.78,.82)], [1,1,0,0]),
           ('car',sample('app/assets/picture3.jpg'),[(.43,.62),(.30,.63),(.20,.65),(.25,.53),(.34,.53),(.44,.53),(.17,.72),(.47,.73),(.24,.64),(.14,.62),(.53,.60)],[1,1,1,0,0,0,0,0,1,1,1]),
           ('nearby-objects',sample('app/assets/picture5.jpg'),[(.57,.50),(.41,.62)],[1,0]),
           ('similar-background',sample('app/assets/picture4.jpg'),[(.56,.56),(.55,.77)],[1,0]),
           ('small-object',sample('MobileSAMv2/test_images/2.jpg'),[(.524,.470),(.56,.49)],[1,0])]
    ddcolor=None
    if args.ddcolor:
        from ddcolor import DDColor,ColorizationPipeline,build_ddcolor_model
        model=build_ddcolor_model(DDColor,model_path=str(ROOT/'modelscope/damo/cv_ddcolor_image-colorization/pytorch_model.pt'),input_size=512,model_size='large')
        ddcolor=ColorizationPipeline(model,input_size=512)
    segmenter=MobileSegmenter(device=args.device)
    records=[]
    def save(name,image):
        if image.ndim==3: image=cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
        if not cv2.imwrite(str(folder/name),image): raise RuntimeError(name)
    for name,image,points,labels in cases:
        if ddcolor is not None and name=='shirt':
            image=cv2.cvtColor(ddcolor.process(cv2.cvtColor(image,cv2.COLOR_RGB2BGR)),cv2.COLOR_BGR2RGB)
        state=RecolorSession.load(image)
        h,w=image.shape[:2]
        start=perf_counter()
        timings=[]
        for (x,y),label in zip(points,labels):
            t=perf_counter();state.select((int(x*w),int(y*h)),bool(label),segmenter)
            timings.append(round(perf_counter()-t,3))
        mask=state.mask.copy()
        state.confirm()
        t=perf_counter();state.apply('#2490d0',100,3);recolor_seconds=perf_counter()-t
        changed=np.any(state.current!=image,axis=2)
        outside=int(np.count_nonzero(changed & ~mask))
        assert outside==0
        assert changed.any()
        before=cv2.cvtColor(image.astype(np.float32)/255,cv2.COLOR_RGB2LAB)[:,:,0]
        after=cv2.cvtColor(state.current.astype(np.float32)/255,cv2.COLOR_RGB2LAB)[:,:,0]
        save(name+'-before.png',image);save(name+'-mask.png',mask.astype(np.uint8)*255)
        save(name+'-after.png',state.current);save(name+'-selection.png',state.overlay())
        records.append(dict(case=name,shape=list(image.shape),points=state.points,labels=state.labels,
            mask_pixels=int(mask.sum()),outside_changed=outside,point_seconds=timings,
            recolor_seconds=round(recolor_seconds,4),luminance_mae=float(np.abs(before[mask]-after[mask]).mean()),
            device=str(segmenter.predictor.device),gpu_allocated_mb=round(torch.cuda.memory_allocated()/1024**2,1),
            gpu_peak_mb=round(torch.cuda.max_memory_allocated()/1024**2,1)))
        print(records[-1],flush=True)
    report=dict(cases=records,ddcolor_loaded=args.ddcolor,checkpoint_sha256=hashlib.sha256(segmenter.checkpoint.read_bytes()).hexdigest(),
        dependencies={p:importlib.metadata.version(p) for p in ['torch','torchvision','gradio','gradio_imageslider','pydantic','huggingface_hub','numpy','opencv-python']})
    (folder/('results-ddcolor.json' if args.ddcolor else 'results.json')).write_text(json.dumps(report,indent=2),encoding='utf-8')
    grid=np.full((len(cases)*245+30,960,3),245,np.uint8)
    for column,title in enumerate(['Before','Mask','After']):
        cv2.putText(grid,title,(column*320+10,22),0,.6,(30,30,30),1)
    for row,(name,_,_,_) in enumerate(cases):
        for column,suffix in enumerate(['before','mask','after']):
            image=cv2.imread(str(folder/f'{name}-{suffix}.png'))
            scale=min(310/image.shape[1],210/image.shape[0])
            image=cv2.resize(image,None,fx=scale,fy=scale)
            x=column*320+(320-image.shape[1])//2
            y=30+row*245
            grid[y:y+image.shape[0],x:x+image.shape[1]]=image
            cv2.putText(grid,name,(column*320+10,y+232),0,.5,(30,30,30),1)
    cv2.imwrite(str(folder/'comparison.png'),grid)


if __name__=='__main__': main()
