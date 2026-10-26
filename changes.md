# Requested changes, not yet made

Queued work. Each entry is written to be picked up without re-deriving
what it means. Delete an entry once it ships.

Nothing queued right now.

---

## Shipped

- **2026-09-23** — "Classes" card retitled **"Add classes"**; its own
  button shortened to "New", since a header and the button inside it
  reading the same words looks like a mistake. The two hints in
  `Track2ScratchTrainingPanel.tsx` that pointed at a card called
  "Classes" now name the new title, so they no longer send anyone
  looking for a card that is not there. Separately, the Assemble panel's
  Classes dropdown now filters out the built-in PPE list, matching the
  five other consumers -- picking it there declared the labels were
  `{0: person, 1: helmet}`, which no training card can then accept, so
  it only led to a dead end. Image `20260923-024656`, verified on the
  host.

- **2026-09-23** — "Save classes" moved onto its own line (it overlapped
  the read-from-dataset toggle); "PPE Safety Judge" renamed to "Training
  and Development" in the wordmark, the sign-in heading and the browser
  tab title; Show/Hide added to "Import a pretrained checkpoint".
  Image `20260923-014931`.

## Still open, needs a decision

- The line under the wordmark still reads **"Jurong Port · case review"**
  (`web/src/components/Layout.tsx:63`). "case review" no longer describes
  the app -- it is training runs, datasets, class lists and a model
  registry -- and it now sits directly under "Training and Development",
  contradicting it. Left alone deliberately: it names the site as well as
  the activity, and only the activity half is wrong.
