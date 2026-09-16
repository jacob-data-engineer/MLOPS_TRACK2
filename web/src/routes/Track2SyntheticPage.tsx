import { Sparkles } from "lucide-react";
import { Card } from "../components/Card";

/**
 * Placeholder -- no generation method has been scoped yet (augmentation
 * vs. a generative model vs. something else entirely is still an open
 * decision). Structure/wiring to follow once that's decided.
 */
export function Track2SyntheticPage() {
  return (
    <div>
      <h1 className="mb-1 flex items-center gap-2 text-xl font-semibold text-ink">
        <Sparkles size={20} className="text-accent" strokeWidth={2} />
        Track2 Synthetic Image Generation
      </h1>
      <p className="mb-6 max-w-3xl text-sm text-ink-secondary">
        Not built yet -- this page is a placeholder until the generation
        approach is scoped.
      </p>

      <Card title="Coming soon">
        <p className="text-sm text-ink-muted">
          This tab will let you generate synthetic training images for the
          Track2 YOLO pipeline. Nothing is wired up yet.
        </p>
      </Card>
    </div>
  );
}
