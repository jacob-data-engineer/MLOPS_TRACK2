# Standing delivery instructions

For completed changes in this repository, without waiting for the user to ask again:

1. Run the checks appropriate to the change and resolve any new failures.
2. Deploy the completed changes to the development app and its worker using the development deployment workflow. Verify the running build and the changed behavior.
3. Commit and push the changes to the GitHub `export` branch.
4. Include a clear commit description explaining what changed, why, and what was verified. Report the development deployment status and commit ID to the user.

These instructions also apply when the user requests a small fix or UI change. Do not leave a completed change only in the local workspace or only pushed to GitHub when development deployment is available.

Preserve unrelated work and existing data. Check for active operations before replacing containers; avoid interrupting an active export. Do not start a training run merely to verify a deployment.

If deployment, verification, or pushing is blocked, explain the specific blocker and clearly state which steps remain incomplete. A later explicit instruction from the user takes precedence.
