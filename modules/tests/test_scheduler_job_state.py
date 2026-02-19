from modules.scheduler import JobScheduler


class _DummyMetrics:
    def __init__(self, state=None):
        self.state = state or {}
        self.logged = []

    def get_job_state(self, job_name):
        return dict(self.state.get(job_name) or {})

    def log_job_run(self, name, status, dur_ms, details):
        self.logged.append((name, status, dur_ms, details))

    def update_job_state(self, name, status, error="", cooldown_until=None):
        self.state[name] = {
            "last_attempt_at": 1,
            "last_success_at": 1 if str(status).lower() == "ok" else None,
            "cooldown_until": cooldown_until,
            "last_error": "" if str(status).lower() == "ok" else str(error),
        }


def test_add_job_respects_cooldown_restore():
    metrics = _DummyMetrics(state={"demo": {"cooldown_until": 9999999999}})
    s = JobScheduler(metrics_store=metrics)
    s.add_job("demo", 10, lambda: None)

    assert s._jobs["demo"]["next_run"] >= 9999999999


def test_add_job_respects_last_success_cadence_restore():
    import time

    now = int(time.time())
    metrics = _DummyMetrics(state={"demo": {"last_success_at": now}})
    s = JobScheduler(metrics_store=metrics)
    s.add_job("demo", 60, lambda: None)

    assert s._jobs["demo"]["next_run"] >= (now + 60)
