"""Small deterministic example: external telemetry matters even when narrative looks safe."""
from .data import generate_dataset
from .pipeline import Config, evaluate


def main():
    cases = generate_dataset(500, 2, 'test')
    hidden = next(c for c in cases if c.scenario == 'hidden_egress')
    benign = next(c for c in cases if c.scenario == 'clean_patch')
    # Force all through screening to isolate the difference between L2 and L3.
    with_logs = Config(use_probe=False, l2_threshold=0, use_telemetry=True)
    without_logs = Config(use_probe=False, l2_threshold=0, use_telemetry=False)
    for case in [benign, hidden]:
        print('\nScenario:', case.scenario)
        print('Narrative:', case.narrative)
        print('Visible tool:', case.tool_action, 'Authorized:', case.allowed_action)
        print('Independent telemetry:', case.telemetry_action or '(none)')
        for name, cfg in [('with external telemetry', with_logs), ('without external telemetry', without_logs)]:
            r = evaluate([case], [0.0], [1.0], cfg)[0]
            print(f'  {name}: queue={r["security_queue"]} urgent={r["urgent_alert"]}')


if __name__ == '__main__':
    main()
