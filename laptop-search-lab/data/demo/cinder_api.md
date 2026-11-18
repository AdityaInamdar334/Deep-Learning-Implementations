# Cinder API changes

The retry-after header contains seconds. Requests without an idempotency key may be duplicated after retries.
