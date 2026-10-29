"""Check Student artifacts even if Kit shutdown loses a Python exit status."""
import argparse
import hashlib
import json
from pathlib import Path


def check(directory, mode, iteration=None):
    root = Path(directory)
    done = json.loads((root/'completion.json').read_text(encoding='utf-8'))
    contract = json.loads((root/'contract.json').read_text(encoding='utf-8'))
    if (done.get('mode')!=mode or not done.get('completed') or not done.get('frozen_weights_verified')
            or not done.get('source_core_verified') or contract.get('schema')!='sfyg_student_v1'):
        raise ValueError('Incomplete Student run or failed frozen/source checks')
    if iteration is not None and done['iteration']!=iteration:
        raise ValueError('Student iteration does not match requested completion')
    if mode=='train':
        n = done['iteration']
        if not (root/f'model_{n}.pt').is_file(): raise FileNotFoundError('Final checkpoint missing')
        bundle = root/'exported'/f'model_{n}'
        manifest = json.loads((bundle/'manifest.json').read_text(encoding='utf-8'))
        if manifest.get('iteration')!=n: raise ValueError('Export iteration mismatch')
        if any(manifest.get(k)!=v for k,v in contract.items()): raise ValueError('Export contract mismatch')
        if manifest.get('policy_sha256')!=hashlib.sha256((bundle/'policy.pt').read_bytes()).hexdigest():
            raise ValueError('Export policy hash mismatch')
        rows = [json.loads(line) for line in (root/'training.jsonl').read_text(encoding='utf-8').splitlines()]
        if not rows or rows[-1]['iteration']!=n: raise ValueError('Missing final training record')
    elif mode=='eval':
        report = json.loads((root/'evaluation.json').read_text(encoding='utf-8'))
        if report.get('contract')!=contract or not report.get('reports'):
            raise ValueError('Missing/mismatched evaluation')
        # A completed diagnostic may legitimately show failed task cases.
        done['required_task_cases_passed'] = report.get('required_task_cases_passed',False)
    else: raise ValueError('Unknown run mode')
    return done


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory')
    parser.add_argument('--mode',required=True,choices=('train','eval'))
    parser.add_argument('--iteration',type=int)
    args = parser.parse_args()
    print(check(args.directory,args.mode,args.iteration))
