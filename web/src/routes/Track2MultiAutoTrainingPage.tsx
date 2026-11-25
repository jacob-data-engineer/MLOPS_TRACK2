import {useEffect, useState} from "react";
import {useSearchParams} from "react-router-dom";
import {Track2ModelsPage} from "./Track2ModelsPage";
import {Track2ExperimentsPage} from "./Track2ExperimentsPage";

export function Track2MultiAutoTrainingPage() {
  const [params, setParams] = useSearchParams();
  const mode = params.get('mode') === 'experiment' ? 'experiment' : 'standard';
  const [visited, setVisited] = useState(() => new Set([mode]));
  useEffect(() => { setVisited(previous => new Set([...previous, mode])); }, [mode]);

  return <div className="space-y-5">
    <fieldset className="rounded border border-hairline p-4">
      <legend className="px-2 font-semibold">Multi-Auto Training mode</legend>
      <div className="flex flex-wrap gap-6">
        {(['standard', 'experiment'] as const).map(value => <label key={value} className="flex items-center gap-2">
          <input type="radio" name="training-mode" value={value} checked={mode === value}
            onChange={() => setParams(previous => {
              const next = new URLSearchParams(previous);
              if (value === 'standard') next.delete('mode');
              else next.set('mode', value);
              return next;
            })}/>
          {value === 'standard' ? 'Standard training' : 'Experiment: baseline + variants'}
        </label>)}
      </div>
    </fieldset>
    {/* Keep visited forms mounted so switching modes preserves unfinished edits.
        Hidden controls are excluded from the tab order and accessibility tree. */}
    <div hidden={mode !== 'standard'}>
      {(mode === 'standard' || visited.has('standard')) && <Track2ModelsPage variant="multi"/>}
    </div>
    <div hidden={mode !== 'experiment'}>
      {(mode === 'experiment' || visited.has('experiment')) && <Track2ExperimentsPage/>}
    </div>
  </div>;
}
