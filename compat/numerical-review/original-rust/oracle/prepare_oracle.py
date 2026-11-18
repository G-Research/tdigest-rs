"""Prepare the exact original-source oracle without compiling or changing fixtures."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import posixpath
import shutil
import subprocess

HERE = Path(__file__).resolve().parent
PIN = '8d9cf1fe82c5c9254e5dc9593f01717e317f4333'
FIXTURE_PIN = '52f6212008111b62fa07374278b850868b9ea77e'
FIXTURE_PATH = 'bindings/python/tests/data/upstream_numerical_golden.json'
GOLDEN_SHA = '517ec82ec124e55920c7049a95f0eb0c9f3e96cf60b4ee7d4457bf31b7a98560'
ORACLE_SHA = '6722a98c481045f408183efd08a575fa95a894f87ed6ec74ac9f994b9486775c'

def sha(data):
    return hashlib.sha256(data).hexdigest()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream-repo', type=Path, required=True)
    parser.add_argument('--modern-repo', type=Path, required=True)
    parser.add_argument('--workdir', type=Path, required=True, help='New disposable directory; must not exist')
    args = parser.parse_args()
    upstream, modern, out = (p.resolve() for p in (args.upstream_repo, args.modern_repo, args.workdir))
    if out.exists() or out.is_relative_to(upstream) or out.is_relative_to(modern):
        parser.error('--workdir must be new and outside both repositories')
    def git(repo, *cmd):
        return subprocess.check_output(['git','-C',str(repo),*cmd])
    assert git(upstream,'rev-parse',PIN).decode().strip() == PIN
    manifest = json.loads((HERE/'source-manifest.json').read_text())
    tree = git(upstream,'ls-tree','-r','-z',PIN).decode().rstrip('\0').split('\0')
    entries = {record.split('\t',1)[1]:record.split('\t',1)[0].split()[0] for record in tree}
    names = list(entries)
    assert set(names) == set(manifest) and len(names) == 35, 'Pinned source file set changed'
    originals = {}
    for name in names:
        rel = PurePosixPath(name)
        assert not rel.is_absolute() and '..' not in rel.parts
        data = git(upstream,'show',PIN+':'+name)
        assert entries[name] in ('100644','100755','120000'), 'Unexpected Git entry mode'
        if entries[name] != '120000':
            assert sha(data) == manifest[name], name
        originals[name] = data
    symlinks = {}
    for name,data in originals.items():
        if entries[name] == '120000':
            link = data.decode()
            target = posixpath.normpath(posixpath.join(posixpath.dirname(name),link))
            assert not posixpath.isabs(link) and target in originals and entries[target] != '120000'
            assert sha(originals[target]) == manifest[name], name
            symlinks[name] = link
    golden = (modern / FIXTURE_PATH).read_bytes()
    assert sha(golden) == GOLDEN_SHA and len(json.loads(golden)['cases']) == 70
    oracle = (HERE/'committed_lock_oracle.rs').read_bytes()
    assert sha(oracle) == ORACLE_SHA
    for name,data in originals.items():
        path=out/'source'/name
        path.parent.mkdir(parents=True,exist_ok=True)
        if name not in symlinks:
            path.write_bytes(data)
            path.chmod(0o755 if entries[name]=='100755' else 0o644)
    for name,target in symlinks.items():
        (out/'source'/name).symlink_to(target)
    for name,digest in manifest.items():
        assert sha((out/'source'/name).read_bytes()) == digest
    (out/'source/examples').mkdir(exist_ok=True)
    (out/'source/examples/committed_lock_oracle.rs').write_bytes(oracle)
    (out/'golden.json').write_bytes(golden)
    shutil.copyfile(HERE/'source-manifest.json',out/'source-manifest.json')
    shutil.copyfile(HERE/'run_oracle.py',out/'run_oracle.py')
    plan={'status':'prepared; no Cargo execution', 'upstream_commit':PIN, 'fixture_source_commit':FIXTURE_PIN, 'original_file_count':35, 'original_symlinks':symlinks, 'original_lock_sha256':manifest['Cargo.lock'], 'oracle_source_sha256':ORACLE_SHA, 'runner_sha256':sha((out/'run_oracle.py').read_bytes()), 'golden_sha256':GOLDEN_SHA, 'case_count':70, 'added_dependencies':[], 'original_manifest_changes':[], 'original_source_changes':[], 'comparison_contract':'Exact finite/infinite/signed-zero bits; NaN classification. Existing fixtures are never regenerated.'}
    (out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    print(json.dumps({'prepared':str(out), 'original_files':35, 'original_lock_sha256':manifest['Cargo.lock'], 'next_command':['python3',str(out/'run_oracle.py'),'--run','--profiles','release','debug']},indent=2))

if __name__ == '__main__':
    main()
