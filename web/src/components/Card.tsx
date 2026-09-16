import type { ReactNode } from "react";

/** The one card container used everywhere: a manifest-page panel -- ruled
 * hairline border, cream surface fill, square-ish corners like a case-file
 * sheet rather than a rounded SaaS tile -- so every panel in the app reads
 * as the same system rather than one-off containers per screen. */
export function Card({
  title,
  caption,
  action,
  children,
}: {
  title?: string;
  caption?: string;
  action?: ReactNode;
  children: ReactNode;
}) {
  return (
    <section className="rounded-sm border border-hairline bg-surface p-5 shadow-[0_1px_0_rgba(36,29,22,0.06)]">
      {(title || action) && (
        <div className="mb-3 flex items-start justify-between gap-4 border-b border-hairline pb-2.5">
          {title && (
            <h3 className="text-base font-semibold tracking-tight text-ink">{title}</h3>
          )}
          {action}
        </div>
      )}
      {caption && <p className="mb-3 text-sm text-ink-secondary">{caption}</p>}
      {children}
    </section>
  );
}
