/** Which cards hold changes that are typed but not yet applied.
 *
 * The toolbar's Save writes a snapshot of the LIVE settings. A card with
 * its own Apply button -- Pull Control is the one that bites -- does not
 * make its form live until that button is pressed, so pressing Save with
 * unapplied changes wrote the OLD settings into the file and reported
 * success. Nothing on screen contradicted it: the card still showed what
 * had been typed.
 *
 * Deliberately a module-level set rather than React state. It is read
 * once, at the moment Save is clicked, by a component that is not an
 * ancestor of the cards -- threading a context through the page for a
 * value nothing renders would be more machinery than the problem needs.
 */
const pending = new Map<string, boolean>();

export function markPending(card: string, isPending: boolean): void {
  pending.set(card, isPending);
}

/** The cards with unapplied changes, in a stable order for the message. */
export function pendingCards(): string[] {
  return [...pending.entries()].filter(([, v]) => v).map(([k]) => k).sort();
}
