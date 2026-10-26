import { AlertCircle, Bell, Loader2, Trash2 } from "lucide-react";
import { useEffect, useState } from "react";
import { Badge } from "../components/Badge";
import { Card } from "../components/Card";
import {
  addTrack2Notification,
  ApiError,
  getTrack2Notifications,
  removeTrack2Notification,
  setTrack2NotificationEvents,
  type Track2NotificationEvent,
  type Track2NotificationSubscription,
} from "../lib/api";

/**
 * Who gets emailed, and about what.
 *
 * The kinds are chosen per subscriber and stored on the SNS subscription
 * itself as a filter policy (see `ppe_pipeline_api.notify`), so SNS does
 * the filtering for every sender and there is no copy here to go stale.
 * Emails that have no kind -- drift review alerts, weights approved, an
 * ordinary pull summary -- still reach every subscriber.
 */

function EventTicks({
  events,
  chosen,
  disabled,
  onChange,
}: {
  events: Track2NotificationEvent[];
  chosen: string[];
  disabled?: boolean;
  onChange: (next: string[]) => void;
}) {
  return (
    <div className="flex flex-col gap-1">
      {events.map((ev) => (
        <label key={ev.key}
               className={`flex items-start gap-2 text-xs ${ev.kiv ? "text-ink-muted" : "text-ink"}`}>
          <input
            type="checkbox"
            className="mt-0.5"
            checked={!ev.kiv && chosen.includes(ev.key)}
            disabled={disabled || ev.kiv}
            onChange={(e) => onChange(e.target.checked
              ? [...chosen, ev.key]
              : chosen.filter((k) => k !== ev.key))}
          />
          <span>
            {ev.label}
            {ev.kiv && (
              <span className="ml-1.5 rounded-sm border border-black/15 bg-page px-1 py-px text-[10px] font-medium uppercase tracking-wide">
                KIV
              </span>
            )}
            {ev.kiv && (
              <span className="block text-[11px] text-ink-muted">
                Kept in view -- not sent yet. It becomes selectable once the
                evaluation step publishes its metrics.
              </span>
            )}
          </span>
        </label>
      ))}
    </div>
  );
}

export function Track2NotificationsPage() {
  const [subscriptions, setSubscriptions] = useState<Track2NotificationSubscription[] | null>(null);
  const [events, setEvents] = useState<Track2NotificationEvent[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [email, setEmail] = useState("");
  const [newEvents, setNewEvents] = useState<string[] | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [saving, setSaving] = useState<string | null>(null);
  const [message, setMessage] = useState<string | null>(null);

  function load() {
    getTrack2Notifications()
      .then((res) => {
        setSubscriptions(res.subscriptions);
        setEvents(res.events);
        // A new subscriber starts with every kind that is actually sent.
        setNewEvents((cur) => cur ?? res.events.filter((e) => !e.kiv).map((e) => e.key));
      })
      .catch((err) => setError(err instanceof ApiError ? err.detail : "Couldn't load subscribers."));
  }

  useEffect(load, []);

  async function handleAdd() {
    if (!email.trim()) return;
    setSubmitting(true);
    setError(null);
    setMessage(null);
    try {
      const res = await addTrack2Notification(email.trim(), newEvents ?? undefined);
      setMessage(res.message);
      setEmail("");
      load();
    } catch (err) {
      setError(err instanceof ApiError ? err.detail : "Subscribe failed.");
    } finally {
      setSubmitting(false);
    }
  }

  async function handleEvents(s: Track2NotificationSubscription, next: string[]) {
    setError(null);
    setSaving(s.email);
    // Shown at once; put back if the save is refused.
    const before = s.events;
    setSubscriptions((all) => all?.map((x) => x.email === s.email ? { ...x, events: next } : x) ?? null);
    try {
      await setTrack2NotificationEvents(s.subscription_arn, next);
    } catch (err) {
      setSubscriptions((all) => all?.map((x) => x.email === s.email ? { ...x, events: before } : x) ?? null);
      setError(err instanceof ApiError ? err.detail : "Could not save that subscriber's choices.");
    } finally {
      setSaving(null);
    }
  }

  async function handleRemove(subscriptionArn: string) {
    setError(null);
    try {
      await removeTrack2Notification(subscriptionArn);
      load();
    } catch (err) {
      setError(err instanceof ApiError ? err.detail : "Remove failed.");
    }
  }

  return (
    <div>
      <h1 className="mb-1 flex items-center gap-2 text-xl font-semibold text-ink">
        <Bell size={20} className="text-accent" strokeWidth={2} />
        Track2 Notifications
      </h1>
      <p className="mb-6 max-w-3xl text-sm text-ink-secondary">
        Email subscribers, and what each one is told about: training
        starting, training ending, and errors. Drift review alerts, weights
        approved and pull summaries still go to everyone.
      </p>

      {error && (
        <div className="mb-6 flex items-center gap-2 rounded-sm border border-critical/30 bg-critical-subtle px-4 py-3 text-sm text-critical">
          <AlertCircle size={16} />
          {error}
        </div>
      )}
      {message && (
        <div className="mb-6 rounded-sm border border-hairline bg-accent-subtle px-4 py-3 text-sm text-ink">
          {message}
        </div>
      )}

      <Card title="Add a subscriber" caption="They'll receive a confirmation email and must click the link before notifications start.">
        <div className="flex gap-2">
          <input
            type="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            placeholder="name@example.com"
            className="w-full max-w-sm rounded-md border border-black/15 bg-page px-3 py-2 text-sm text-ink"
          />
          <button
            type="button"
            onClick={handleAdd}
            disabled={submitting || !email.trim()}
            className="flex shrink-0 items-center gap-1.5 rounded-md bg-accent px-3 py-2 text-sm font-medium text-white disabled:opacity-40"
          >
            {submitting && <Loader2 size={14} className="animate-spin" />}
            Subscribe
          </button>
        </div>
        {events.length > 0 && newEvents && (
          <div className="mt-3">
            <p className="mb-1 text-xs font-medium text-ink">Send them</p>
            <EventTicks events={events} chosen={newEvents} onChange={setNewEvents} />
          </div>
        )}
      </Card>

      <div className="mt-5">
        <Card title="Subscribers">
          {!subscriptions && !error && (
            <div className="flex items-center gap-2 text-sm text-ink-muted">
              <Loader2 size={15} className="animate-spin" />
              Loading...
            </div>
          )}
          {subscriptions && subscriptions.length === 0 && (
            <p className="text-sm text-ink-muted">No subscribers yet.</p>
          )}
          {subscriptions && subscriptions.length > 0 && (
            <div className="flex flex-col gap-2">
              {subscriptions.map((s) => (
                <div
                  /* Not subscription_arn: every unconfirmed subscriber has
                     the same literal "PendingConfirmation" there, so two of
                     them collided on one React key. One address can only be
                     subscribed to a topic once. */
                  key={s.email}
                  className="rounded-sm border border-hairline bg-page p-3"
                >
                  <div className="flex items-center justify-between gap-3">
                    <div className="flex items-center gap-2">
                      <span className="text-sm text-ink">{s.email}</span>
                      <Badge tone={s.confirmed ? "confirmed" : "pending"} label={s.confirmed ? "confirmed" : "pending confirmation"} />
                      {saving === s.email && <Loader2 size={13} className="animate-spin text-ink-muted" />}
                    </div>
                    {s.confirmed ? (
                      <button
                        type="button"
                        onClick={() => handleRemove(s.subscription_arn)}
                        className="flex items-center gap-1.5 rounded-sm border border-black/15 px-2.5 py-1.5 text-xs font-medium text-ink-secondary hover:bg-black/[0.04]"
                      >
                        <Trash2 size={13} />
                        Remove
                      </button>
                    ) : (
                      /* There is nothing to remove until they confirm: AWS has
                         not issued a subscription ARN yet, and offers no way to
                         delete one that does not exist. Showing the button
                         anyway meant a click, a 500, and "Remove failed."
                         with no explanation -- every time, forever. */
                      <span className="text-xs text-ink-muted">
                        expires on its own in 3 days
                      </span>
                    )}
                  </div>
                  <div className="mt-2 border-t border-hairline pt-2">
                    {s.events ? (
                      <EventTicks
                        events={events}
                        chosen={s.events}
                        disabled={saving === s.email}
                        onChange={(next) => void handleEvents(s, next)}
                      />
                    ) : (
                      <p className="text-xs text-ink-muted">
                        {s.confirmed
                          ? "Could not read this subscriber's choices."
                          : "Their choices were set when they were added, and can be changed once they confirm."}
                      </p>
                    )}
                  </div>
                </div>
              ))}
            </div>
          )}
        </Card>
      </div>
    </div>
  );
}
