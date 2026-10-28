"""Check artifacts without Isaac; guards against Kit shutdown masking exit codes."""
import argparse
import hashlib
import json
from pathlib import Path

from sfyg_teacher_evaluation import baseline_gate


def check(path,mode):
    path=Path(path)
    done=json.loads((path/'completion.json').read_text(encoding='utf-8'))
    if (done.get('mode')!=mode or not done.get('completed') or not done.get('frozen_weights_verified')
            or not done.get('source_core_verified')):
        raise ValueError('Missing successful completion/frozen-source verification')
    if mode=='smoke':
        audit=json.loads((path/'interface_audit.json').read_text(encoding='utf-8'))
        if not audit.get('passed'): raise ValueError('Interface audit did not pass')
    if mode=='baseline':
        report=json.loads((path/'evaluation.json').read_text(encoding='utf-8'))
        if (report.get('mode')!='baseline' or not report.get('frozen_weights_verified')
                or not baseline_gate(report.get('reports',[]))['passed']):
            raise ValueError('T0 motion/reward gate failed; teacher training will not start')
    if mode=='train':
        run=json.loads((path/'run.json').read_text(encoding='utf-8'))
        count=run['arguments']['max_iterations']
        if not (path/f'model_{count}.pt').is_file(): raise ValueError('Final teacher checkpoint missing')
        export=path/'exported'/f'model_{count}'
        manifest=json.loads((export/'manifest.json').read_text(encoding='utf-8'))
        if hashlib.sha256((export/'policy.pt').read_bytes()).hexdigest()!=manifest['policy_sha256']:
            raise ValueError('Export policy hash mismatch')
    return done


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('path')
    parser.add_argument('--mode',choices=('smoke','baseline','train'),required=True)
    args=parser.parse_args()
    check(args.path,args.mode)
    print(f'[CHECK] {args.mode}: artifact checks passed: {args.path}')
