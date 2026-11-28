# Named development baseline registry permissions

The dev host role `track2-yolo-review-host` uses the inline policy
`track2-named-dev-baselines` from `dev_baseline_registry_policy.json`.
It allows creation, discovery, inspection and deletion of versions only in the
`baseline-track2-dev-*` namespace. It does not grant production registry access,
endpoint deployment, training execution, or deletion of package groups.

Apply from an authorized AWS administration session:

```powershell
aws iam put-role-policy --role-name track2-yolo-review-host --policy-name track2-named-dev-baselines --policy-document file://scripts/dev_baseline_registry_policy.json
```

Imports use the existing comparison registration service: preserve a separate
artifact, create a version in the label-specific registry, and index the group
in SQL for discovery. The exact user label is the package description.

On 28 November 2026, FAKECAD IT BASELINE was imported from the legacy development
baseline registry version 2 into
`baseline-track2-dev-fakecad-it-baseline-7c17e39e5d`, version 1. Saved comparison
settings referencing that legacy version were updated. Historical run ARNs and
results were retained; only their display label was standardized. No training
was launched and no old registry version was deleted.
