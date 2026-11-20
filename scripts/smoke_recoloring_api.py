"""Exercise the running Gradio app, including SelectData and session state."""
import json
import argparse
from pathlib import Path
import uuid
import requests
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
URL='http://127.0.0.1:7860'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--remove',action='store_true')
    args=parser.parse_args()
    prefix='remove' if args.remove else 'recolor'
    session=uuid.uuid4().hex
    config=requests.get(URL+'/config',timeout=10).json()
    dependencies={d['api_name']:d for d in config['dependencies']}
    image_path=ROOT/'third_party/MobileSAM/app/assets/picture3.jpg'
    with image_path.open('rb') as image:
        response=requests.post(URL+'/upload',files={'files':(image_path.name,image,'image/jpeg')},timeout=30)
    response.raise_for_status()
    path=response.json()[0]
    def call(name,data,event=None):
        payload=dict(data=data,session_hash=session,event_data=event)
        response=requests.post(URL+'/api/'+name,json=payload,timeout=180)
        if response.status_code!=200: raise RuntimeError(response.text)
        result=response.json()
        if result.get('error'): raise RuntimeError(result)
        print(name, json.dumps(result.get('data',[])[-1:],ensure_ascii=True),flush=True)
        return result['data']
    loaded=call(prefix+'_load',[{'path':path,'meta':{'_type':'gradio.FileData'}},*([3] if args.remove else [])])
    selected=call(prefix+'_select',[None,'+ Thêm vùng',None],{'index':[963,930],'value':None})
    # Each action receives the same State; a real SelectData is supplied to the handler.
    call(prefix+'_confirm',[None,None])
    applied=call(prefix+'_apply',[None,None,*(['telea',3,3] if args.remove else ['#d83030',85,3])])
    output_path=applied[1]['path']
    original=cv2.imread(loaded[1]['path'])
    output=cv2.imread(output_path)
    assert np.any(original!=output)
    selected_image=cv2.imread(selected[2]['path'])
    mask=np.any(selected_image!=original,axis=2)
    assert not np.any(output[~mask]!=original[~mask])
    undo=call(prefix+'_undo',[None,None])
    np.testing.assert_array_equal(cv2.imread(undo[1]['path']),original)
    call(prefix+'_reset',[None,None])
    report=dict(result='passed',actions=['upload','load','select','confirm','apply','download','undo','reset'],
                outside_changed=0,dependencies=len(dependencies))
    # Download endpoint must serve the actual lossless result.
    download=requests.get(URL+'/file='+output_path,timeout=30)
    download.raise_for_status()
    folder=ROOT/('reports/removal' if args.remove else 'reports/recoloring')
    folder.mkdir(parents=True,exist_ok=True)
    (folder/'api-after.png').write_bytes(download.content)
    (folder/'api-smoke.json').write_text(json.dumps(report,indent=2),encoding='utf-8')


if __name__=='__main__': main()
