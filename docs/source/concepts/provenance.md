(provenance)=
# Provenance and artifacts

Scarf records each analysis result as a logical artifact: a Zarr group with a
stable reference, a completion flag, and a provenance record. Downstream steps
take artifact references (or published `AssayState` pointers) as inputs, so the
store can reuse completed work when the same operation is requested again.

## Artifact identity

Each artifact has a random 64-character hex `artifact_id`. The ID is not a hash
of the result. It only names the group on disk.

Reuse is decided by **canonical provenance equality**. Two calls reuse the same
complete artifact when these three fields match after canonical serialization:

- `operation`: the named step (for example `run_normalization`)
- `parameters`: analysis parameters that define the scientific result
- `inputs`: upstream selections and artifact references the step consumed

`execution_options` are stored on the artifact but are **not** part of identity.
Argument roles are operation-specific. `local_cache` and `batch_size` are
reduction execution options, while `n_centroids` is an identity parameter
because it changes the fitted result. Changing only values recorded under
`execution_options` still reuses a matching complete artifact unless you force
a rebuild.

## Completion contract

While a writer is in progress, the group exists with `complete=False`. Treat an
incomplete artifact as untrusted. Readers that require a finished result reject
incomplete groups.

A complete artifact has:

- `artifact_id`, `kind`, `provenance`, `execution_options`, and `complete=True`
- the arrays and attributes required by that kind

Use the store APIs rather than parsing Zarr attrs by hand:

```python
refs = ds.list_artifacts(kind="normalized")
status = ds.inspect_artifact(refs[0])
status.complete
status.operation
status.parameters
status.inputs
status.execution_options

group = ds.load_artifact(refs[0])  # requires complete=True
```

`list_artifacts` defaults to the assay scope for the default assay. Pass
`scope="datastore"` for store-level artifacts, and `complete_only=True` to skip
incomplete groups.

## Reuse and invalidation

Identical provenance reuses the newest matching complete artifact. Pass
`invalidate_cache=True` on the producing method to skip reuse and write a new
artifact with a new ID. Older complete artifacts remain in the store until you
delete them yourself; invalidation does not prune history.

Typical pattern:

```python
normalized = ds.run_normalization(feat_key="hvgs")
again = ds.run_normalization(feat_key="hvgs")
assert again == normalized

forced = ds.run_normalization(feat_key="hvgs", invalidate_cache=True)
assert forced != normalized
```

Changing an identity parameter (for example PCA `dims` or neighbor `k`) creates
a different provenance record and therefore a different artifact. Upstream
artifacts whose provenance still matches are reused; only the changed step and
its dependents need new results.

## Metadata links

Cell and feature columns written by analysis steps often store a
`source_artifact` attribute pointing at the artifact that produced them. Marker
search keeps an index under `{assay}/markers` whose `artifacts` attr maps
`{cell_key}__{group_key}` slots to `marker_table` artifact refs; the table
payload lives under `{assay}/artifacts/marker_table/{id}`.

These links are the practical way to answer "which artifact wrote this column?"
without scanning the whole store.

## Walking lineage

Scarf does not yet ship a lineage graph helper. Walk inputs yourself from
`inspect_artifact`:

```python
def walk_inputs(ds, ref, seen=None):
    seen = seen if seen is not None else set()
    if ref.artifact_id in seen:
        return
    seen.add(ref.artifact_id)
    status = ds.inspect_artifact(ref)
    print(ref.kind, ref.artifact_id[:12], status.operation)
    for value in (status.inputs or {}).values():
        if isinstance(value, dict) and value.get("type") == "artifact":
            walk_inputs(ds, scarf.ArtifactRef.from_dict(value), seen)

state = ds.get_assay_state("RNA")
if state is not None and state.connectivity_map is not None:
    walk_inputs(ds, state.connectivity_map)
```

Start from a published state pointer (`get_assay_state`), a column's
`source_artifact`, or a ref returned by `pipeline.run` / an atomic method.

## What provenance does not do

- It does not content-hash matrix payloads for identity (inputs are referenced,
  not re-fingerprinted on every lookup beyond what the step recorded).
- It does not delete superseded artifacts.
- It does not replace experimental notebooks; it makes store-backed results
  inspectable and reusable across sessions.

For a hands-on walkthrough, see {doc}`../tutorials/provenance_and_reuse`.
For how pipeline, atomic steps, and `AssayState` fit together, see
{doc}`graph_and_state`.
