"""Fill benchmark LPIPS with one isolated model load per clean reference."""
import argparse
import csv
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from demo.metrics_backend import lpips_files


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('result',type=Path)
    args=parser.parse_args()
    payload=json.loads(args.result.read_text(encoding='utf-8'))
    rows=payload['rows']
    for reference in sorted({row['reference'] for row in rows}):
        selected=[row for row in rows if row['reference']==reference]
        print('Scoring',len(selected),'pairs against',reference,flush=True)
        scores=lpips_files(reference,[row['image'] for row in selected])
        for row,score in zip(selected,scores):
            row['lpips']=score
    args.result.write_text(json.dumps(payload,indent=2),encoding='utf-8')
    with args.result.with_suffix('.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print('LPIPS saved',flush=True)


if __name__=='__main__':
    main()
