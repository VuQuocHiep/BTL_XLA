"""Deterministic paired corruptions; one source, not a generalization dataset."""
import cv2
import numpy as np


def make_cases(side=256):
    from skimage.data import astronaut

    color = cv2.resize(astronaut(), (side,side), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(color,cv2.COLOR_RGB2GRAY)
    clean = cv2.cvtColor(gray,cv2.COLOR_GRAY2RGB)
    rng = np.random.default_rng(20261102)
    cases = {}
    zero = np.zeros((side,side),np.uint8)
    cases['clean'] = (clean.copy(),zero.copy())
    noisy = np.clip(gray.astype(float)+rng.normal(0,10,gray.shape),0,255).astype(np.uint8)
    cases['noise'] = (cv2.cvtColor(noisy,cv2.COLOR_GRAY2RGB),zero.copy())
    for name, thickness in [('curved_scratch',2),('wide_damage',max(7,side//24))]:
        mask = zero.copy()
        points = np.array([[side//5,side//10],[side//3,side//3],[side//4,side*3//5],[side*2//5,side*9//10]],np.int32)
        cv2.polylines(mask,[points],False,255,thickness)
        if name == 'wide_damage':
            cv2.rectangle(mask,(side*3//5,side//3),(side*3//4,side//2),255,-1)
        damaged = clean.copy()
        damaged[mask>0] = 245
        cases[name] = (damaged,mask)
    mask = zero.copy()
    for x,y in rng.integers(8,side-8,(30,2)):
        cv2.circle(mask,(int(x),int(y)),1,255,-1)
    damaged=clean.copy(); damaged[mask>0]=250
    cases['dust']=(damaged,mask)
    faded=np.clip(clean.astype(float)*0.55+55,0,255).astype(np.uint8)
    cases['faded']=(faded,zero.copy())
    cases['blur']=(cv2.GaussianBlur(clean,(0,0),1.2),zero.copy())
    mixed=cases['noise'][0].copy(); mask=cases['curved_scratch'][1].copy();mixed[mask>0]=245
    cases['mixed']=(mixed,mask)
    return color,clean,cases
