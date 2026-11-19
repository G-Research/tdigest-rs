// Used by the standalone release.yml workflow, with read-only GitHub access.
module.exports = async function prepare({github, context, core}) {
  const {owner, repo} = context.repo;
  const repository = `${owner}/${repo}`;
  const manual = context.eventName === 'workflow_dispatch';
  const runId = String(manual ? context.payload.inputs.run_id : context.payload.workflow_run.id);
  if (!/^\d+$/.test(runId) || !Number.isSafeInteger(Number(runId)) || Number(runId) <= 0) {
    throw new Error('Expected a positive CI run ID');
  }
  if (context.ref !== `refs/heads/${context.payload.repository.default_branch}`) {
    throw new Error('Release must run from the default branch');
  }

  const {data: run} = await github.rest.actions.getWorkflowRun({owner, repo, run_id: Number(runId)});
  if (run.repository.full_name !== repository || run.head_repository.full_name !== repository ||
      run.path !== '.github/workflows/ci.yml' || run.event !== 'push' || run.status !== 'completed') {
    throw new Error('Expected a completed push run of this repository\'s CI workflow');
  }
  // Only the manual recovery route can accept the old CI's failed publishing job.
  if (run.conclusion !== 'success' && !(manual && run.conclusion === 'failure')) {
    throw new Error('CI must succeed before automatic publishing');
  }
  const tag = run.head_branch;
  if (!/^v\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?$/.test(tag)) {
    throw new Error('Expected a version tag CI run');
  }
  // Require an actual tag, not a similarly named branch. getCommit peels annotated tags.
  await github.rest.git.getRef({owner, repo, ref: `tags/${tag}`});
  const {data: commit} = await github.rest.repos.getCommit({owner, repo, ref: `refs/tags/${tag}`});
  if (commit.sha !== run.head_sha) {
    throw new Error('The release tag no longer matches the tested commit');
  }

  const jobs = await github.paginate(github.rest.actions.listJobsForWorkflowRun, {
    owner, repo, run_id: run.id, filter: 'latest', per_page: 100,
  });
  const required = jobs.filter(job => job.name === 'All required checks done');
  if (required.length !== 1 || required[0].conclusion !== 'success') {
    throw new Error('The required CI gate must have succeeded, not been skipped');
  }
  const oldPublishJobs = new Set(['release / Publish to PyPI', 'release / Publish GitHub release']);
  if (jobs.some(job => job.conclusion !== 'success' && !(manual && oldPublishJobs.has(job.name)))) {
    throw new Error('A build or test job did not succeed');
  }

  const artifacts = (await github.paginate(github.rest.actions.listWorkflowRunArtifacts, {
    owner, repo, run_id: run.id, per_page: 100,
  })).filter(artifact => artifact.name.startsWith('python-package-'));
  const builds = jobs.filter(job => job.name.startsWith('python / Build ('));
  if (builds.length === 0 || artifacts.length !== builds.length + 1 ||
      new Set(artifacts.map(artifact => artifact.name)).size !== artifacts.length ||
      !artifacts.some(artifact => artifact.name === 'python-package-sdist') ||
      artifacts.some(artifact => artifact.expired)) {
    throw new Error('Expected an unexpired wheel artifact per build and the checked source archive');
  }
  let releaseExists = false;
  try {
    await github.rest.repos.getReleaseByTag({owner, repo, tag});
    releaseExists = true;
  } catch (error) {
    if (error.status !== 404) throw error;
  }

  core.setOutput('run_id', String(run.id));
  core.setOutput('tag', tag);
  core.setOutput('artifact_ids', artifacts.map(artifact => artifact.id).join(','));
  core.setOutput('release_exists', String(releaseExists));
  await core.summary
    .addHeading(`Publish ${tag}`)
    .addLink('Tested CI run', run.html_url)
    .addRaw(`\n\nCommit: ${run.head_sha}\n\nArtifacts: ${artifacts.map(artifact => artifact.name).join(', ')}\n`)
    .write();
};
