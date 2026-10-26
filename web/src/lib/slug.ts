/** Turning the name somebody types into the identifier a registry group
 * is built from.
 *
 * There used to be two fields, Slug and Name, side by side. One of them
 * silently decided where every version of that Class would be filed for
 * the rest of its life, and the other was a label; which was which was
 * not obvious, and the pair read as the same question asked twice.
 *
 * So there is one field now, and this is what it becomes. ONE
 * implementation, in its own module, because two forms create Classes --
 * "Define a new Class" and the import card, when a checkpoint's class
 * names match nothing that exists -- and two derivations that disagreed
 * would file the same typed name under two different group names.
 *
 * The rules are the server's, not invented here: `upsert_detector`
 * requires `[a-z0-9]([a-z0-9-]{0,30})`, because the result is
 * interpolated into `yolo{generation}{size}-{slug}-detector`. Anything
 * this produces has to pass that, or the only feedback is a 422 naming a
 * field the operator can no longer see.
 */

/** The longest a slug may be. The server's own limit: 1 + 30. */
export const SLUG_MAX = 31;

/** `Vehicle detection!` -> `vehicle-detection`. "" when nothing survives.
 *
 * Empty is a real answer and must stay one: a name of "!!!" has no
 * identifier in it, and inventing something like "class-1" would file a
 * model under a name nobody chose and nobody could find again.
 */
export function slugFromName(name: string): string {
  return (name || "")
    .toLowerCase()
    // Any run of anything else becomes a single hyphen, so "Fall zone /
    // vehicle" does not become "fall-zone--vehicle".
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, SLUG_MAX)
    // Truncation can leave a trailing hyphen, which the server rejects.
    .replace(/-+$/, "");
}

/** Why this name cannot become a Class, or "". Phrased for somebody who
 * never saw the slug it is really talking about. */
export function slugProblem(name: string): string {
  const typed = (name || "").trim();
  if (!typed) return "";
  if (!slugFromName(typed)) {
    return "Use at least one letter or digit -- the name becomes part of "
      + "the registry group every version of this Class is filed under.";
  }
  return "";
}

/** The registry group a Class of this slug registers into, at one
 * generation and size. Mirrors `group_pattern` in `upsert_detector`;
 * shown so the consequence of the name is visible while it is typed
 * rather than discovered afterwards in the registry. */
export function groupNameFor(slug: string, generation: string, size: string): string {
  return `yolo${generation}${size}-${slug}-detector`;
}


/** The prefix the server puts on a version registered as a SEED -- one
 * whose weights detect something other than the Class it is filed under,
 * because it is about to be retrained and the head rebuilt.
 *
 * Mirrors `SEED_DESCRIPTION_PREFIX` in `track2_shared.py`. Read off the
 * description rather than from a field of its own, because that is what
 * the registry scan already carries and what the AWS console shows; a
 * test asserts the two strings stay equal. */
export const SEED_PREFIX = "SEED --";

/** Is this version a seed rather than a trained model?
 *
 * NOT "is it version 1". A v1 produced by a training run is a real
 * model, and labelling it a seed would be a lie in the one column whose
 * job is saying what a version is. */
export function isSeedVersion(description: string): boolean {
  return (description || "").trimStart().startsWith(SEED_PREFIX);
}


/** How a registry status reads on screen.
 *
 * `Approved` is SageMaker's word for "this version may be used", and it
 * is the value the service stores -- unchanged, because
 * `export_weights_to_folder` and the approval rule both key on it. What
 * it MEANS here is that the version is ready to deploy or train from,
 * so that is what it says.
 *
 * `PendingManualApproval` is shortened for the same reason the status
 * column is narrow; a seed is handled before this is ever reached.
 */
export function statusLabel(status: string): string {
  if (status === "Approved") return "Ready";
  if (status === "PendingManualApproval") return "Pending";
  return status;
}
