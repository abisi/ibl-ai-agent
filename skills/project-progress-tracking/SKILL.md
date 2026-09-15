---
name: project-progress-tracking
description: Keep a project's TODO.md (or equivalent running log) in sync with actual project state throughout a long-running session, so work is always resumable after a restart or a new session -- not just at the end.
---

# Project Progress Tracking

## Use this skill when

- Working in a project directory that has a `TODO.md` (or similarly-named running log/status file), and the session spans multiple non-trivial actions, decisions, or background tasks over an extended period.
- Especially relevant when background compute (a scheduler, a remote sweep, a long-running job) is in flight, since progress and methodology can change while no one is actively watching the file.

## The problem this prevents

A `TODO.md` that is only updated when the user happens to ask, or only at the very end of a session, silently drifts out of sync with what has actually happened: methodology changes, queue reorderings, newly-discovered gaps, and completed/pending status all live only in the conversation, not on disk. If the session ends (context runs out, the PC restarts, a new session starts cold), that undocumented state is effectively lost, and "resuming the project" becomes archaeology instead of reading a file. This was caught concretely in the `ssl-whisker-hitmiss-timeresolved-decoding` project (2026-09-15): several consequential changes -- a decoding-estimator switch, a null-control policy change, a reordered remote job queue -- had happened but were not yet in `TODO.md` when the user asked "is the TODO list complete?".

## The rule

Roughly every 5 minutes of active work on a project with a running log -- in practice, at natural checkpoints rather than a literal timer: after completing a task, after a background stage finishes, after any consequential methodology/code/scope decision, or whenever a status check is requested -- pause and ask: **does the project's running log still accurately describe (a) what has been decided, (b) what is currently running or complete, and (c) what remains to be done?** If it has drifted, update it immediately, in the project's existing running-log style (dated entries, append rather than rewrite history), before moving on to further work. Do not wait for the user to notice the gap and ask.

This is a documentation cadence, not an automation request: nothing here requires a scheduled wakeup or background timer running independently of the conversation. It means treating "keep the log current" as a standing background obligation during the session, the same way test coverage or error handling would be, rather than a favor done only when explicitly asked.

## How to apply

1. When picking up a project mid-session (or cold, after a restart/new session), read the running log first -- but verify anything load-bearing against the actual current state (running processes, latest result files, current code) before treating it as ground truth, since it may itself be stale.
2. When a consequential change happens (a methodology switch, a schema/estimator change, a scope decision, a queue reordering, a newly-found gap), write it to the log promptly -- do not let several such changes accumulate undocumented, waiting for a convenient moment.
3. When asked for a status check, treat it as a trigger to also verify the log is current, not just to report status conversationally -- if the two have diverged, fix the log as part of answering.
4. For a project whose real-time state lives partly outside this session (e.g. a background job on a remote host), the log should say exactly where the authoritative live state can be re-derived from (a script path, a log file, a command) -- so the log stays useful even after it goes stale, rather than becoming actively misleading.

## Quality gates

- Reject ending a significant piece of work, or a long stretch of a session, without confirming the project's running log reflects what actually happened -- a stale log defeats its own purpose.
- Reject treating a running log as complete just because it exists and has recent-looking entries -- check whether the entries actually cover the most recent consequential changes, not just whether the file was touched recently.
- Reject writing a log update that describes intent ("we will do X") when X has already happened -- update the log to reflect the actual current state, not a stale plan.
