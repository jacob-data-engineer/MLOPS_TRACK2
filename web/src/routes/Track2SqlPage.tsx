import { AlertCircle, Copy, Database, Loader2, Play, Table2 } from "lucide-react";
import { useEffect, useRef, useState, type KeyboardEvent } from "react";
import { Card } from "../components/Card";
import { S3FolderField } from "../components/S3FolderField";
import {
  ApiError,
  getTrack2Buckets,
  getTrack2PullControl,
  getTrack2SqlDatabases,
  getTrack2SqlSchema,
  runTrack2Sql,
  saveTrack2PullConfig,
  type Track2SqlDatabase,
  type Track2SqlResult,
  type Track2SqlTable,
} from "../lib/api";

/** Rows drawn in the table. The images list below it is never cut. */
const ROWS_SHOWN = 500;

/** TRACK2 SQL DATABASE QUERY (2026-10-28).
 *
 * Type SQL against one of the Postgres databases and get back the images
 * the rows name. The query runs in a small function inside the database's
 * own network, as a read-only user, in a read-only transaction, one
 * statement at a time -- nothing typed here can change the database.
 *
 * Which cells are images is read from the values (s3:// URIs, S3 links,
 * paths ending in an image extension), not the column names. The bucket
 * and folder narrow that list; leave them blank to see every image.
 *
 * The database, bucket, folder and query are saved like every other
 * card's settings, so a settings file brings them back.
 */
export function Track2SqlPage() {
  const [dbs, setDbs] = useState<Track2SqlDatabase[]>([]);
  const [database, setDatabase] = useState("");
  const [buckets, setBuckets] = useState<string[]>([]);
  const [bucket, setBucket] = useState("");
  const [folder, setFolder] = useState("");
  const [sql, setSql] = useState("");
  const [tables, setTables] = useState<Track2SqlTable[] | null>(null);
  const [openTable, setOpenTable] = useState("");
  const [result, setResult] = useState<Track2SqlResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [schemaBusy, setSchemaBusy] = useState(false);
  const [error, setError] = useState("");
  const [copied, setCopied] = useState(false);
  // The writer below must not run before the saved settings are read, or
  // it would save the empty defaults over them.
  const [settingsRead, setSettingsRead] = useState(false);
  const editor = useRef<HTMLTextAreaElement | null>(null);

  useEffect(() => {
    getTrack2SqlDatabases()
      .then((res) => setDbs(res.databases))
      .catch(() => setDbs([]));
    getTrack2Buckets()
      .then((res) => setBuckets(res.buckets.filter((b) => b.readable).map((b) => b.name)))
      .catch(() => undefined);
    getTrack2PullControl()
      .then((status) => {
        const saved = status.config.sql_explorer || {};
        if (saved.database) setDatabase(saved.database);
        if (saved.bucket) setBucket(saved.bucket);
        if (saved.folder) setFolder(saved.folder);
        if (saved.sql) setSql(saved.sql);
      })
      .catch(() => undefined)
      .finally(() => setSettingsRead(true));
  }, []);

  // First database by default, once the list is in and nothing was saved.
  useEffect(() => {
    if (settingsRead && !database && dbs.length) setDatabase(dbs[0].key);
  }, [settingsRead, database, dbs]);

  useEffect(() => {
    if (!settingsRead) return;
    const timer = window.setTimeout(() => {
      void saveTrack2PullConfig({ sql_explorer: { database, bucket, folder, sql } })
        .catch(() => undefined);
    }, 700);
    return () => window.clearTimeout(timer);
  }, [settingsRead, database, bucket, folder, sql]);

  // A different database has different tables.
  useEffect(() => {
    setTables(null);
    setOpenTable("");
  }, [database]);

  const chosen = dbs.find((d) => d.key === database);

  function loadTables() {
    if (!database) return;
    setSchemaBusy(true);
    setError("");
    getTrack2SqlSchema(database)
      .then((res) => setTables(res.tables))
      .catch((err) => setError(err instanceof ApiError ? err.detail : "Could not read the tables."))
      .finally(() => setSchemaBusy(false));
  }

  function run() {
    if (!database || !sql.trim() || busy) return;
    setBusy(true);
    setError("");
    setCopied(false);
    runTrack2Sql(database, sql, bucket, folder)
      .then(setResult)
      .catch((err) => {
        setResult(null);
        setError(err instanceof ApiError ? err.detail : "The query failed.");
      })
      .finally(() => setBusy(false));
  }

  function onEditorKey(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
      e.preventDefault();
      run();
    }
  }

  /** Puts text at the cursor, so clicking a column name writes it there. */
  function insert(text: string) {
    const box = editor.current;
    if (!box) {
      setSql((current) => current + text);
      return;
    }
    const start = box.selectionStart ?? sql.length;
    const end = box.selectionEnd ?? sql.length;
    const next = sql.slice(0, start) + text + sql.slice(end);
    setSql(next);
    window.requestAnimationFrame(() => {
      box.focus();
      box.selectionStart = box.selectionEnd = start + text.length;
    });
  }

  function copyImages() {
    if (!result) return;
    void navigator.clipboard
      .writeText(result.images.map((image) => image.uri).join("\n"))
      .then(() => setCopied(true))
      .catch(() => undefined);
  }

  return (
    <div>
      <h1 className="mb-1 flex items-center gap-2 text-xl font-semibold text-ink">
        <Database size={20} className="text-accent" strokeWidth={2} />
        Track2 SQL Database Query
      </h1>
      <p className="mb-4 max-w-3xl text-sm text-ink-secondary">
        Query a Postgres database and get back the images its rows name. Read
        only: each query runs as a read-only user, in a read-only transaction,
        one statement at a time.
      </p>

      <div className="flex flex-col gap-5">
        <Card title="Where to look" collapsible={false}
              caption="The database to query, and the bucket and folder the images must be in. Leave the bucket and folder blank to keep every image the query finds.">
          <div className="grid gap-3 md:grid-cols-2">
            <label className="block text-xs text-ink-muted">
              Database
              <select
                value={database}
                onChange={(e) => { setDatabase(e.target.value); setResult(null); }}
                className="mt-1 w-full rounded border border-black/15 bg-page px-2 py-1.5 text-sm text-ink"
              >
                {dbs.map((d) => (
                  <option key={d.key} value={d.key}>
                    {d.label}{d.available ? "" : ` -- ${d.status}`}
                  </option>
                ))}
              </select>
              {chosen && !chosen.available && (
                <span className="mt-1 block text-critical">
                  This database is {chosen.status}. Queries will not get an answer until it is running.
                </span>
              )}
            </label>
            <label className="block text-xs text-ink-muted">
              S3 bucket
              <select
                value={bucket}
                onChange={(e) => { setBucket(e.target.value); setFolder(""); }}
                className="mt-1 w-full rounded border border-black/15 bg-page px-2 py-1.5 text-sm text-ink"
              >
                <option value="">Any bucket</option>
                {buckets.map((name) => (
                  <option key={name} value={name}>{name}</option>
                ))}
              </select>
            </label>
          </div>
          <div className="mt-3">
            {bucket ? (
              <S3FolderField
                label="Folder"
                hint="only images under this folder; blank for the whole bucket"
                value={folder}
                onChange={setFolder}
                bucket={bucket}
              />
            ) : (
              <p className="text-xs text-ink-muted">Pick a bucket to narrow the images to one of its folders.</p>
            )}
          </div>
        </Card>

        <Card title="Query" collapsible={false}
              caption="Ctrl+Enter runs it. Click a table to start a query on it, or a column to type its name at the cursor.">
          <div className="grid gap-4 md:grid-cols-[16rem_1fr]">
            <div className="max-h-96 overflow-y-auto rounded border border-hairline bg-page p-2 text-xs">
              <div className="mb-2 flex items-center justify-between gap-2">
                <span className="font-medium text-ink">Tables</span>
                <button
                  type="button"
                  onClick={loadTables}
                  disabled={!database || schemaBusy}
                  className="rounded border border-black/15 px-2 py-0.5 text-ink disabled:opacity-50"
                >
                  {schemaBusy ? "Reading..." : tables ? "Reload" : "Load tables"}
                </button>
              </div>
              {tables === null && !schemaBusy && (
                <p className="text-ink-muted">Not loaded yet.</p>
              )}
              {tables?.length === 0 && <p className="text-ink-muted">No tables this user can read.</p>}
              {tables?.map((table) => (
                <div key={table.name} className="mb-1">
                  <div className="flex items-center gap-1">
                    <button
                      type="button"
                      onClick={() => setOpenTable(openTable === table.name ? "" : table.name)}
                      className="text-ink-muted"
                      aria-label={openTable === table.name ? "Hide columns" : "Show columns"}
                    >
                      {openTable === table.name ? "▾" : "▸"}
                    </button>
                    <button
                      type="button"
                      onClick={() => setSql(`SELECT *\nFROM ${table.name}\nLIMIT 100`)}
                      className="tabular truncate text-left text-ink hover:underline"
                      title={`Start a query on ${table.name}`}
                    >
                      <Table2 size={12} className="mr-1 inline" />
                      {table.name}
                    </button>
                  </div>
                  {openTable === table.name && (
                    <ul className="ml-5 mt-0.5">
                      {table.columns.map((column) => (
                        <li key={column.name}>
                          <button
                            type="button"
                            onClick={() => insert(column.name)}
                            className="tabular text-left text-ink-secondary hover:underline"
                          >
                            {column.name} <span className="text-ink-muted">{column.type}</span>
                          </button>
                        </li>
                      ))}
                    </ul>
                  )}
                </div>
              ))}
            </div>

            <div>
              <textarea
                ref={editor}
                value={sql}
                onChange={(e) => setSql(e.target.value)}
                onKeyDown={onEditorKey}
                spellCheck={false}
                rows={10}
                placeholder={"SELECT image_key\nFROM some_table\nWHERE ...\nLIMIT 100"}
                className="tabular w-full rounded border border-black/15 bg-page px-3 py-2 font-mono text-sm text-ink"
              />
              <div className="mt-2 flex flex-wrap items-center gap-3">
                <button
                  type="button"
                  onClick={run}
                  disabled={busy || !database || !sql.trim()}
                  className="inline-flex items-center gap-1.5 rounded-md bg-accent px-3 py-1.5 text-sm font-medium text-white hover:bg-accent-hover disabled:opacity-40"
                >
                  {busy ? <Loader2 size={14} className="animate-spin" /> : <Play size={14} />}
                  {busy ? "Running..." : "Run query"}
                </button>
                {result && (
                  <span className="tabular text-xs text-ink-muted">
                    {result.rows.length.toLocaleString()} row{result.rows.length === 1 ? "" : "s"}
                    {result.truncated && " (cut short: the answer was too large -- add a LIMIT or select fewer columns)"}
                  </span>
                )}
              </div>
              {error && (
                <p className="mt-2 flex items-start gap-1.5 text-sm text-critical">
                  <AlertCircle size={15} className="mt-0.5 shrink-0" />
                  <span className="whitespace-pre-wrap">{error}</span>
                </p>
              )}
            </div>
          </div>
        </Card>

        {result && (
          <Card
            title={`Images (${result.images.length.toLocaleString()})`}
            collapsible={false}
            caption={bucket
              ? `Images the rows name, in ${bucket}${folder ? `/${folder.replace(/\/+$/, "")}` : ""}.`
              : "Every image the rows name, in any bucket."}
          >
            {result.images.length === 0 ? (
              <p className="text-sm text-ink-muted">
                No images in this result{bucket ? " inside the chosen bucket and folder" : ""}. An image is a
                value that is an s3:// address, an S3 link, or a path ending in .jpg, .png and the like.
              </p>
            ) : (
              <>
                <div className="mb-2 flex items-center gap-3">
                  <button
                    type="button"
                    onClick={copyImages}
                    className="inline-flex items-center gap-1.5 rounded-md border border-black/15 bg-page px-2.5 py-1 text-xs font-medium text-ink"
                  >
                    <Copy size={13} />
                    Copy the list
                  </button>
                  {copied && <span className="text-xs text-ink-muted">Copied.</span>}
                </div>
                <ol className="tabular max-h-96 list-decimal overflow-y-auto rounded border border-hairline bg-page py-1 pl-10 pr-2 text-xs text-ink">
                  {result.images.map((image) => (
                    <li key={image.uri} className="truncate py-0.5" title={`row ${image.row + 1}, column ${image.column}`}>
                      {image.uri}
                    </li>
                  ))}
                </ol>
              </>
            )}
          </Card>
        )}

        {result && result.columns.length > 0 && (
          <Card
            title="Rows"
            collapsible={false}
            caption={result.rows.length > ROWS_SHOWN
              ? `The first ${ROWS_SHOWN.toLocaleString()} of ${result.rows.length.toLocaleString()} rows. The images above come from all of them.`
              : undefined}
          >
            <div className="max-h-[32rem] overflow-auto rounded border border-hairline">
              <table className="tabular min-w-full text-xs">
                <thead className="sticky top-0 bg-surface">
                  <tr>
                    {result.columns.map((column, i) => (
                      <th key={`${column}-${i}`} className="border-b border-hairline px-2 py-1 text-left font-medium text-ink">
                        {column}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {result.rows.slice(0, ROWS_SHOWN).map((row, r) => (
                    <tr key={r} className="odd:bg-black/[0.02]">
                      {row.map((cell, c) => (
                        <td key={c} className="max-w-xs truncate border-b border-hairline/50 px-2 py-0.5 text-ink-secondary"
                            title={cellText(cell)}>
                          {cellText(cell)}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        )}
      </div>
    </div>
  );
}

function cellText(value: unknown): string {
  if (value === null || value === undefined) return "NULL";
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
}
