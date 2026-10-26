import { useEffect, useRef, type ReactNode } from "react";

/**
 * The one confirmation dialog.
 *
 * `window.confirm` was what this app used, and it is wrong for anything
 * that matters: it cannot carry a sentence of consequence, it cannot
 * show what the server just said, it cannot offer a second, harsher
 * option, and it looks like a browser error rather than part of the
 * page. Worst of all, it forces a yes/no on information the operator
 * does not have yet -- which is how deleting a Model Name ended up as a
 * confirm, then a red line at the top of the card, then a separate
 * button nowhere near the row that was clicked.
 *
 * So: one dialog, holding the question, the consequence, whatever the
 * server answered, and every choice available. Nothing about the action
 * appears anywhere else on the page.
 *
 * DEFAULTS TO CANCEL. The cancel button takes focus, Escape closes, and
 * a click on the backdrop closes -- so every reflex that means "not
 * this" does the safe thing. A destructive confirm is never the default
 * action and never focused.
 */
export function ConfirmDialog({
  open,
  title,
  body,
  confirmLabel = "Confirm",
  cancelLabel = "Cancel",
  destructive = false,
  busy = false,
  onConfirm,
  onCancel,
}: {
  open: boolean;
  title: string;
  /** The consequence, in the reader's terms. A dialog that only repeats
   * the button it came from has told them nothing they did not know. */
  body: ReactNode;
  confirmLabel?: string;
  cancelLabel?: string;
  /** Reds the confirm button. For anything that cannot be undone. */
  destructive?: boolean;
  /** Mid-request: both buttons lock so a double-click cannot send it
   * twice, and the label says what is happening. */
  busy?: boolean;
  onConfirm: () => void;
  onCancel: () => void;
}) {
  const cancelRef = useRef<HTMLButtonElement | null>(null);

  useEffect(() => {
    if (!open) return;
    // Focus lands on Cancel, not Confirm: Enter on a freshly opened
    // dialog must not delete anything.
    cancelRef.current?.focus();

    function onKey(event: KeyboardEvent) {
      if (event.key === "Escape" && !busy) onCancel();
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, busy, onCancel]);

  if (!open) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4"
      // A click on the backdrop means "not this". Guarded while busy so
      // a request in flight is not abandoned halfway with the dialog
      // gone and nothing saying what happened.
      onClick={() => { if (!busy) onCancel(); }}
    >
      <div
        role="dialog"
        aria-modal="true"
        aria-label={title}
        className="w-full max-w-md rounded-sm border border-hairline bg-surface p-5 shadow-lg"
        // Or a click anywhere inside would close it through the backdrop.
        onClick={(event) => event.stopPropagation()}
      >
        <h2 className="text-sm font-semibold text-ink">{title}</h2>
        <div className="mt-2 text-xs text-ink-secondary">{body}</div>

        <div className="mt-4 flex justify-end gap-2">
          <button
            ref={cancelRef}
            type="button"
            disabled={busy}
            onClick={onCancel}
            className="rounded-md border border-black/15 px-3 py-1.5 text-xs font-medium text-ink-secondary disabled:opacity-40"
          >
            {cancelLabel}
          </button>
          <button
            type="button"
            disabled={busy}
            onClick={onConfirm}
            className={
              destructive
                ? "rounded-md border border-critical bg-critical px-3 py-1.5 text-xs font-medium text-white disabled:opacity-40"
                : "rounded-md bg-accent px-3 py-1.5 text-xs font-medium text-white disabled:opacity-40"
            }
          >
            {busy ? "Working..." : confirmLabel}
          </button>
        </div>
      </div>
    </div>
  );
}
