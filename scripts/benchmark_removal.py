"""Record actual MobileSAM + installed OpenCV / existing LaMa results."""
import json
from pathlib import Path
import sys
from time import perf_counter

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
import torch
from demo.object_segmentation import shared_segmenter
from demo.object_removal import RemovalSession, removal_mask, remove_object
from demo.restoration_backends import available_models


def main():
    torch.set_num_threads(4)
    folder=ROOT/'reports/removal'
    folder.mkdir(parents=True,exist_ok=True)
    samples=ROOT/'third_party/MobileSAM'
    cases=[('small-object','MobileSAMv2/test_images/2.jpg',[(.524,.470),(.56,.49)],[1,0]),
           ('dog','app/assets/picture4.jpg',[(.56,.56),(.55,.77)],[1,0])]
    records=[]
    segmenter=shared_segmenter()
    def save(name,image):
        if image.ndim==3:image=cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
        assert cv2.imwrite(str(folder/name),image)
    for name,path,points,labels in cases:
        image=cv2.cvtColor(cv2.imread(str(samples/path)),cv2.COLOR_BGR2RGB)
        state=RemovalSession.load(image)
        t=perf_counter()
        for (x,y),label in zip(points,labels):
            state.select((int(x*image.shape[1]),int(y*image.shape[0])),bool(label),segmenter)
        selection_seconds=perf_counter()-t
        mask=removal_mask(state.mask,3)
        save(name+'-before.png',image);save(name+'-selection.png',state.overlay())
        save(name+'-mask.png',mask.astype(np.uint8)*255)
        for backend in ['telea','ns']+(['lama'] if available_models()['lama'] else []):
            t=perf_counter()
            output=remove_object(image,state.mask,backend,3,3)
            outside=int(np.count_nonzero(np.any(output!=image,axis=2)&~mask))
            assert outside==0 and np.any(output!=image)
            save(name+'-'+backend+'-after.png',output)
            record=dict(case=name,backend=backend,shape=list(image.shape),mask_pixels=int(mask.sum()),
                outside_changed=outside,selection_seconds=round(selection_seconds,3),
                removal_seconds=round(perf_counter()-t,3))
            records.append(record);print(record,flush=True)
    (folder/'results.json').write_text(json.dumps(dict(opencv=cv2.__version__,cases=records),indent=2),encoding='utf-8')


if __name__=='__main__':main()
