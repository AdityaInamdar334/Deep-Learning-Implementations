# Webhook Delivery: Design Draft (intentionally incomplete)

## Problem
Users need a webhook whenever a job finishes.

## Goals and Non-Goals
Goal: emit a JSON POST within a few seconds of job completion. Non-goal: support complex subscriptions.

## Architecture
On job completion, call the webhook URL stored on the customer's account. Retry on failure.

## Failure Modes
Use exponential backoff. TODO: choose when to stop.

## Implementation Plan
Add a webhook sender to the existing backend. Endpoint location TBD.

## Acceptance Tests
Check that a successful delivery returns a 2xx response.
