# Webhook Delivery: Revised Teaching Example

## Problem
A completed background job should notify its owning account without blocking the job transaction. Current clients poll the job-status API. The event payload contains an event ID, job ID, status, and completion timestamp, but not private job content.

## Goals and Non-Goals
Goal: persist a delivery request and deliver at least once with a bounded retry policy. Non-goals: exactly-once networking semantics, arbitrary payload scripting, global ordering across accounts.

## Architecture
After committing the job status update, an outbox row records (event_id, account_id, destination_id, payload_hash, created_at). A worker claims an outbox row, reads the active destination URL, signs the canonical request body using an account-scoped webhook secret, and attempts HTTPS delivery. Receivers must deduplicate using event_id. Rate limiting is per destination and overall. Destination URLs are prevalidated at registration, DNS is rechecked at connection time, and private/non-public address ranges are blocked; redirects are disabled. Worker has no access to production administrative credentials.

## Failure Modes
Retries use a maximum of five attempts, exponential backoff capped at ten minutes with jitter. Non-2xx status, timeout after five seconds, and network exceptions count as failures. After the limit, mark delivery dead-letter and expose operator-visible retry controls; manual retries retain the same event_id. Log statuses and correlation IDs, not secrets or payload bodies. Destination deletion cancels pending deliveries. Capacity is bounded by worker concurrency and a per-account quota.

## Implementation Plan
Paths below are proposed, not asserted as existing: `backend/webhooks/outbox.py` defines OutboxRecord; `backend/webhooks/worker.py` implements claim_next and deliver_once; `backend/webhooks/security.py` signs payload and validates destinations; `backend/migrations/20261107_webhook_outbox.sql` creates the table; `backend/tests/test_webhooks.py` exercises behavior. Before implementation, inspect repo structure and replace these suggestions with verified paths. Database transaction and outbox claim implementation must be chosen to match actual SQL dialect, with a documented migration and rollback plan.

## Acceptance Tests
Test durable enqueue after job commit, duplicate event replay with stable event_id, timeout retry, maximum attempts, dead-letter visibility, URL validation for loopback/private IP, redirect prohibition, signature validity, secret rotation overlap, cancellation on destination deletion, and no secret leakage in logs. Include controlled load test for worker backpressure and verify authorization on destination configuration.
