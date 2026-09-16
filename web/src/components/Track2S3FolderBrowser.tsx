import { ChevronRight, File, Folder, FolderPlus, Search } from "lucide-react";
import { useMemo, useState } from "react";

/**
 * Windows-Explorer-style S3 prefix browser: a clickable breadcrumb address
 * bar (jump straight to any ancestor, no repeated ".." clicks), a search
 * box that filters the folders currently in view, and a checkbox beside
 * each folder for selection -- separate from navigation, so picking a
 * folder never requires first navigating INTO it (that's what caused the
 * malformed/duplicated paths in the previous click-to-navigate-then-
 * confirm design: navigating and selecting were the same action, easy to
 * fire twice against a path that had already changed underneath it).
 *
 * Only one folder can be the actual selection at a time (`selected` is a
 * single path), so checking a box implicitly un-checks whichever one was
 * checked before -- it behaves like a radio group despite looking like
 * checkboxes, which is what "selected folders get a checkbox" means once
 * there's only ever one real destination value to send the API.
 */
export function Track2S3FolderBrowser({
  prefix,
  folders,
  onNavigate,
  onSelect,
  selected,
  allowCreate = false,
  files,
  onSelectFile,
  selectedFile,
}: {
  prefix: string;
  folders: string[];
  onNavigate: (prefix: string) => void;
  onSelect?: (prefix: string) => void;
  selected?: string;
  /** S3 has no real "create folder" operation -- a folder is just the
   * common prefix of whatever gets written under it. Enabling this lets
   * the caller select (and navigate into) a name that doesn't exist as a
   * folder yet, which becomes real the moment something is actually
   * written there (an export, an upload). Only meaningful where the
   * destination is allowed to be new -- Export's destination folder, not
   * a dataset/base-model folder that has to already contain real data. */
  allowCreate?: boolean;
  /** Files at this exact level (e.g. .pt/.tar.gz checkpoints) -- an
   * entirely separate selection from the folder one above (a file is
   * never also a valid "destination folder"), so it's its own prop pair
   * rather than overloading `selected`/`onSelect`. Omit to hide files
   * entirely, matching every existing caller that only ever wanted
   * folders. */
  files?: string[];
  onSelectFile?: (key: string) => void;
  selectedFile?: string;
}) {
  const [search, setSearch] = useState("");
  const [newFolderName, setNewFolderName] = useState("");

  // S3 keys never have a leading slash -- stripping both ends (not just
  // trailing) keeps every join below from ever producing "/name" for a
  // root-level click, which used to silently 404 one level down.
  const trimmed = prefix.replace(/^\/+|\/+$/g, "");
  const selectedTrimmed = selected?.replace(/^\/+|\/+$/g, "");
  const segments = trimmed ? trimmed.split("/") : [];
  const isCurrentFolderSelected = selectedTrimmed !== undefined && selectedTrimmed !== "" && selectedTrimmed === trimmed;

  function joinPath(base: string, name: string): string {
    return base ? `${base}/${name}` : name;
  }

  function pathUpTo(index: number): string {
    return segments.slice(0, index + 1).join("/");
  }

  const filteredFolders = useMemo(() => {
    const q = search.trim().toLowerCase();
    if (!q) return folders;
    return folders.filter((name) => name.toLowerCase().includes(q));
  }, [folders, search]);

  const filteredFiles = useMemo(() => {
    const q = search.trim().toLowerCase();
    if (!files) return [];
    if (!q) return files;
    return files.filter((name) => name.toLowerCase().includes(q));
  }, [files, search]);

  function handleCreate() {
    const name = newFolderName.trim();
    if (!name) return;
    const fullPath = joinPath(trimmed, name);
    onNavigate(fullPath);
    onSelect?.(fullPath);
    setNewFolderName("");
  }

  return (
    <div className="rounded-sm border border-hairline bg-page">
      <div className="flex flex-wrap items-center gap-x-1 gap-y-0.5 border-b border-hairline bg-surface px-3 py-2 text-xs">
        {onSelect && (
          <input
            type="checkbox"
            className="mr-1.5 shrink-0"
            checked={isCurrentFolderSelected}
            onChange={(e) => onSelect(e.target.checked ? trimmed : "")}
            title="Select this exact folder (the one you're currently browsing)"
          />
        )}
        <button
          type="button"
          onClick={() => onNavigate("")}
          className="tabular text-ink-secondary hover:text-ink hover:underline"
        >
          bucket root
        </button>
        {segments.map((seg, i) => (
          <span key={i} className="flex items-center gap-1">
            <span className="text-ink-muted">/</span>
            <button
              type="button"
              onClick={() => onNavigate(pathUpTo(i))}
              className="tabular text-ink-secondary hover:text-ink hover:underline"
            >
              {seg}
            </button>
          </span>
        ))}
      </div>

      <div className="flex items-center gap-2 border-b border-hairline px-3 py-2">
        <Search size={13} className="shrink-0 text-ink-muted" />
        <input
          type="text"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          placeholder="Search this view..."
          className="w-full bg-transparent text-xs text-ink outline-none placeholder:text-ink-muted"
        />
      </div>

      {allowCreate && (
        <div className="flex items-center gap-2 border-b border-hairline px-3 py-2">
          <FolderPlus size={13} className="shrink-0 text-ink-muted" />
          <input
            type="text"
            value={newFolderName}
            onChange={(e) => setNewFolderName(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") handleCreate();
            }}
            placeholder="New folder name..."
            className="w-full bg-transparent text-xs text-ink outline-none placeholder:text-ink-muted"
          />
          <button
            type="button"
            onClick={handleCreate}
            disabled={!newFolderName.trim()}
            className="shrink-0 rounded-sm border border-black/15 px-2 py-1 text-xs font-medium text-ink-secondary disabled:opacity-40"
          >
            Create &amp; use
          </button>
        </div>
      )}

      <div className="max-h-56 overflow-y-auto">
        {filteredFolders.length === 0 && filteredFiles.length === 0 && (
          <p className="px-3 py-2 text-xs text-ink-muted">
            {folders.length === 0 && (files?.length ?? 0) === 0 ? "Nothing here." : "Nothing matches your search."}
          </p>
        )}
        {filteredFolders.map((name) => {
          const fullPath = joinPath(trimmed, name);
          const isChecked = selectedTrimmed !== undefined && selectedTrimmed === fullPath;
          return (
            <div key={name} className="flex items-center gap-2 px-3 py-1.5 text-sm text-ink hover:bg-black/[0.04]">
              {onSelect && (
                <input
                  type="checkbox"
                  className="shrink-0"
                  checked={isChecked}
                  onChange={(e) => onSelect(e.target.checked ? fullPath : "")}
                />
              )}
              <button
                type="button"
                onClick={() => onNavigate(fullPath)}
                className="flex min-w-0 flex-1 items-center gap-2 text-left"
                title={`Open ${fullPath}`}
              >
                <Folder size={14} className="shrink-0 text-ink-muted" />
                <span className="truncate">{name}</span>
                <ChevronRight size={13} className="ml-auto shrink-0 text-ink-muted" />
              </button>
            </div>
          );
        })}
        {filteredFiles.map((name) => {
          const fullKey = joinPath(trimmed, name);
          const isChecked = selectedFile === fullKey;
          return (
            <label
              key={name}
              className="flex items-center gap-2 px-3 py-1.5 text-sm text-ink hover:bg-black/[0.04]"
            >
              <input
                type="checkbox"
                className="shrink-0"
                checked={isChecked}
                onChange={(e) => onSelectFile?.(e.target.checked ? fullKey : "")}
              />
              <File size={14} className="shrink-0 text-ink-muted" />
              <span className="tabular truncate">{name}</span>
            </label>
          );
        })}
      </div>
    </div>
  );
}
