const assert = require('node:assert/strict');
const {readFileSync, readdirSync} = require('node:fs');
const path = require('node:path');
const {test} = require('node:test');
const prepare = require('./prepare_pypi_release.cjs');

const workflows = path.join(__dirname, '..', '.github', 'workflows');

test('PyPI publishing runs directly in the trusted release.yml workflow', () => {
  const release = readFileSync(path.join(workflows, 'release.yml'), 'utf8');
  assert.match(release, /uses: pypa\/gh-action-pypi-publish@/);
  assert.doesNotMatch(release, /workflow_call:/,
    'A reusable publisher signs the calling workflow identity, not release.yml');
  for (const filename of readdirSync(workflows).filter(name => name.endsWith('.yml'))) {
    assert.doesNotMatch(readFileSync(path.join(workflows, filename), 'utf8'),
      /uses:\s*\.\/\.github\/workflows\/release\.yml/,
      `${filename} must not call the PyPI publisher as a reusable workflow`);
  }
});

function fixture(manual = false) {
  const state = {
    context: {
      repo: {owner: 'G-Research', repo: 'tdigest-rs'},
      eventName: manual ? 'workflow_dispatch' : 'workflow_run',
      ref: 'refs/heads/main',
      payload: {repository: {default_branch: 'main'}, inputs: {run_id: '123'}, workflow_run: {id: 123}},
    },
    run: {
      id: 123, repository: {full_name: 'G-Research/tdigest-rs'},
      head_repository: {full_name: 'G-Research/tdigest-rs'},
      path: '.github/workflows/ci.yml', event: 'push', status: 'completed', conclusion: 'success',
      head_branch: 'v1.0.0', head_sha: 'a'.repeat(40), html_url: 'https://github.com/run/123',
    },
    tagSha: 'a'.repeat(40),
    jobs: [
      {name: 'All required checks done', conclusion: 'success'},
      {name: 'rust / Test (ubuntu-latest)', conclusion: 'success'},
      {name: 'python / Build (ubuntu-latest, x86_64-unknown-linux-gnu)', conclusion: 'success'},
    ],
    artifacts: [
      {id: 11, name: 'python-package-x86_64-unknown-linux-gnu', expired: false},
      {id: 12, name: 'python-package-sdist', expired: false},
      {id: 13, name: 'quality-results', expired: false},
    ],
    releaseExists: false,
    tagExists: true,
    outputs: {},
    requests: [],
  };
  const actions = {
    getWorkflowRun: async args => {
      assert.deepEqual(args, {owner: 'G-Research', repo: 'tdigest-rs', run_id: 123});
      return {data: state.run};
    },
    listJobsForWorkflowRun: Symbol('jobs'),
    listWorkflowRunArtifacts: Symbol('artifacts'),
  };
  const github = {
    rest: {
      actions,
      git: {getRef: async args => {
        assert.equal(args.ref, `tags/${state.run.head_branch}`);
        if (!state.tagExists) throw Object.assign(new Error('Tag not found'), {status: 404});
        return {data: {object: {sha: state.tagSha}}};
      }},
      repos: {
        getCommit: async args => {
          assert.equal(args.ref, `refs/tags/${state.run.head_branch}`);
          return {data: {sha: state.tagSha}};
        },
        getReleaseByTag: async () => {
          if (!state.releaseExists) throw Object.assign(new Error('Not found'), {status: 404});
          return {data: {id: 456}};
        },
      },
    },
    paginate: async (method, args) => {
      state.requests.push(args);
      assert.equal(args.run_id, 123);
      assert.equal(args.per_page, 100);
      if (method === actions.listJobsForWorkflowRun) {
        assert.equal(args.filter, 'latest');
        return state.jobs;
      }
      assert.equal(method, actions.listWorkflowRunArtifacts);
      return state.artifacts;
    },
  };
  const summary = {addHeading() {return this;}, addLink() {return this;}, addRaw() {return this;}, async write() {}};
  state.execute = () => prepare({github, context: state.context,
    core: {setOutput: (key, value) => {state.outputs[key] = value;}, summary}});
  return state;
}

test('successful tag CI selects only its package artifacts', async () => {
  const state = fixture();
  await state.execute();
  assert.deepEqual(state.outputs, {
    run_id: '123', tag: 'v1.0.0', artifact_ids: '11,12', release_exists: 'false',
  });
});

test('manual recovery accepts the failed old publisher after all tests passed', async () => {
  const state = fixture(true);
  state.run.conclusion = 'failure';
  state.jobs.push({name: 'release / Publish to PyPI', conclusion: 'failure'});
  state.jobs.push({name: 'release / Publish GitHub release', conclusion: 'success'});
  state.releaseExists = true;
  await state.execute();
  assert.equal(state.outputs.release_exists, 'true');
  assert.equal(state.outputs.artifact_ids, '11,12');
});

for (const [name, change, error] of [
  ['a feature-branch dispatch', s => {s.context.ref = 'refs/heads/feature';}, /default branch/],
  ['a non-numeric run ID', s => {s.context.payload.inputs.run_id = '123; echo bad';}, /positive CI run ID/],
  ['an unsafe integer run ID', s => {s.context.payload.inputs.run_id = '9007199254740993';}, /positive CI run ID/],
  ['a fork run', s => {s.run.head_repository.full_name = 'someone/tdigest-rs';}, /completed push run/],
  ['a different repository', s => {s.run.repository.full_name = 'someone/tdigest-rs';}, /completed push run/],
  ['a pull request run', s => {s.run.event = 'pull_request';}, /completed push run/],
  ['a different workflow', s => {s.run.path = '.github/workflows/other.yml';}, /completed push run/],
  ['an unfinished run', s => {s.run.status = 'in_progress';}, /completed push run/],
  ['a cancelled run', s => {s.run.conclusion = 'cancelled';}, /CI must succeed/],
  ['a branch build', s => {s.run.head_branch = 'main';}, /version tag/],
  ['a version branch without a tag', s => {s.tagExists = false;}, /Tag not found/],
  ['a moved tag', s => {s.tagSha = 'b'.repeat(40);}, /no longer matches/],
  ['a skipped required gate', s => {s.jobs[0].conclusion = 'skipped';}, /required CI gate/],
  ['a missing required gate', s => {s.jobs.shift();}, /required CI gate/],
  ['a duplicate required gate', s => {s.jobs.push({...s.jobs[0]});}, /required CI gate/],
  ['a failed test during recovery', s => {s.jobs[1].conclusion = 'failure';}, /build or test job/],
  ['a skipped test during recovery', s => {s.jobs[1].conclusion = 'skipped';}, /build or test job/],
  ['an expired package', s => {s.artifacts[0].expired = true;}, /unexpired wheel/],
  ['a missing source archive', s => {s.artifacts.splice(1, 1);}, /unexpired wheel/],
  ['a missing wheel', s => {s.artifacts.shift();}, /unexpired wheel/],
  ['a missing second platform', s => {s.jobs.push({name: 'python / Build (windows-latest, x86_64-pc-windows-msvc)', conclusion: 'success'});}, /unexpired wheel/],
]) {
  test(`rejects ${name} before exposing artifacts to the publisher`, async () => {
    const state = fixture(true);
    change(state);
    await assert.rejects(state.execute, error);
    assert.deepEqual(state.outputs, {});
  });
}

test('automatic publishing rejects a failed source run', async () => {
  const state = fixture();
  state.run.conclusion = 'failure';
  state.jobs.push({name: 'release / Publish to PyPI', conclusion: 'failure'});
  await assert.rejects(state.execute, /CI must succeed/);
  assert.deepEqual(state.outputs, {});
});
