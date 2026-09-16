import { AlertCircle, Bell, Loader2, Trash2 } from "lucide-react";
import { useEffect, useState } from "react";
import { Badge } from "../components/Badge";
import { Card } from "../components/Card";
import {
  addTrack2Notification,
  ApiError,
  getTrack2Notifications,
  removeTrack2Notification,
  type Track2NotificationSubscription,
} from "../lib/api";

export function Track2NotificationsPage() {
  const [subscriptions, setSubscriptions] = useState<Track2NotificationSubscription[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [email, setEmail] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [message, setMessage] = useState<string | null>(null);

  function load() {
    getTrack2Notifications()
      .then((res) => setSubscriptions(res.subscriptions))
      .catch((err) => setError(err instanceof ApiError ? err.detail : "Couldn't load subscribers."));
  }

  useEffect(load, []);

  async function handleAdd() {
    if (!email.trim()) return;
    setSubmitting(true);
    setError(null);
    setMessage(null);
    try {
      const res = await addTrack2Notification(email.trim());
      setMessage(res.message);
      setEmail("");
      load();
    } catch (err) {
      setError(err instanceof ApiError ? err.detail : "Subscribe failed.");
    } finally {
      setSubmitting(false);
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
        Email subscribers for training pipeline status, drift-triggered
        review alerts, and weights-approved notifications.
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
                  key={s.subscription_arn}
                  className="flex items-center justify-between gap-3 rounded-sm border border-hairline bg-page p-3"
                >
                  <div className="flex items-center gap-2">
                    <span className="text-sm text-ink">{s.email}</span>
                    <Badge tone={s.confirmed ? "confirmed" : "pending"} label={s.confirmed ? "confirmed" : "pending confirmation"} />
                  </div>
                  <button
                    type="button"
                    onClick={() => handleRemove(s.subscription_arn)}
                    className="flex items-center gap-1.5 rounded-sm border border-black/15 px-2.5 py-1.5 text-xs font-medium text-ink-secondary hover:bg-black/[0.04]"
                  >
                    <Trash2 size={13} />
                    Remove
                  </button>
                </div>
              ))}
            </div>
          )}
        </Card>
      </div>
    </div>
  );
}
