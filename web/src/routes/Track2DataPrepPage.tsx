import { Database } from "lucide-react";
import { useState } from "react";
import { SwitchedOff } from "../components/SwitchedOff";
import { Track2AutoTrainingPanel } from "../components/Track2AutoTrainingPanel";
import { Track2DatasetAssemblePanel } from "../components/Track2DatasetAssemblePanel";
import { Track2GoldenSetPanel } from "../components/Track2GoldenSetPanel";
import { Track2PinnedBar } from "../components/Track2PinnedBar";
import { Track2PullControlPanel } from "../components/Track2PullControlPanel";
import { Track2PullQueuePanel } from "../components/Track2PullQueuePanel";

/** DATA PREPARATION: everything that decides what a run trains on.
 *
 * Split out of Multi-Auto Training on 2026-10-28. This tab turns files in
 * the bucket into the sources of a dataset -- folders picked by hand, the
 * golden set, the pulled week folders and the retention policy over them.
 * The Training tab decides what to train and trains it; at Add to Queue
 * and on the schedule the chain freezes these sources into a dataset
 * version, which is the only thing a run is ever handed.
 *
 * Nothing on this page talks to the Training tab directly. Each card saves
 * its choices into the settings record, which is what the chain and the
 * Training tab read, and posts "data-sources-changed" on the notice board
 * (lib/bus.ts) so a Training tab open elsewhere refreshes its summary.
 */
export function Track2DataPrepPage() {
  // Whether the dataset is picked by hand on the Assemble card, which
  // switches Pull Control, Auto Training and Retention off.
  //
  // Declared here with the rest of the state, not beside the cards that
  // read it. A `const` that reads a `useState` declared BELOW it throws on
  // every render and unmounts the whole tree -- the page went white that
  // way once already, and `tsc` does not catch it inside a callback.
  //
  // The Assemble card restores the saved tick into this on load.
  const [pickFoldersByHand, setPickFoldersByHand] = useState(false);

  return (
    <div>
      {/* The same pinned strip as Training: the pull mode in it is the
          one switch for the automatic chain, which starts on this tab. */}
      <div className="sticky top-0 z-30 -mx-4 -mt-6 mb-4 border-b border-hairline bg-page px-4 pb-2 pt-6 md:-mx-8 md:-mt-8 md:px-8 md:pt-8">
        <h1 className="mb-1 flex items-center gap-2 text-xl font-semibold text-ink">
          <Database size={20} className="text-accent" strokeWidth={2} />
          Track2 Data Preparation
        </h1>
        <p className="mb-2 max-w-3xl text-sm text-ink-secondary">
          What a training run learns from: folders picked by hand, the golden
          set, the pulled week folders and the retention policy over them.
          Training happens on the Multi-Auto Training tab.
        </p>
        <Track2PinnedBar scope="dp" />
      </div>

      <div className="flex flex-col gap-5">
        {/* Pick the data by hand, instead of collecting it week by week.
            FIRST, because it is the first question -- where does this
            run's data come from -- and the cards underneath are the other
            answer to it. Ticking "Use these folders only" switches them off
            rather than letting both decide. The Model Name it names classes
            after is the one saved on the Training tab. */}
        <Track2DatasetAssemblePanel
          multiSource
          only={pickFoldersByHand}
          onOnlyChange={setPickFoldersByHand}
        />

        {/* Divides a set somebody curated by hand. Ahead of the pull: what
            belongs in a golden set is a judgement made on images you
            already have, so it owes the pull nothing. */}
        <Track2GoldenSetPanel />

        {/* Upstream of the weeks below: pull control is what puts files
            into the week folders that Auto Training then discovers. Not
            wrapped in `SwitchedOff` -- the tick that turns it back on is in
            its own heading, so it dims its own body instead. */}
        <Track2PullControlPanel
          only={pickFoldersByHand}
          onOnlyChange={setPickFoldersByHand}
        />

        {/* The week folders, with Retention inside: retention decides how
            much of each older week a run takes. */}
        <SwitchedOff
          off={pickFoldersByHand}
          reason={<>The run is using the folders picked above. Retention
            lives inside this card, so it is off with it.</>}
        >
          <Track2AutoTrainingPanel />
        </SwitchedOff>

        {/* Sweeps waiting their turn. Here with the Pull Control card whose
            settings they queue; the training queue is on the Training tab. */}
        <Track2PullQueuePanel />
      </div>
    </div>
  );
}
