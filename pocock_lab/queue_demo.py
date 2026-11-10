"""Original example. Verify the cancellation contract and demonstrate a controlled regression."""
from dataclasses import dataclass

@dataclass
class Run:
    state: str = "queued"

    def cancel(self):
        if self.state == "queued":
            self.state = "cancelled"
        return self.state

    def claim(self):
        if self.state != "queued":
            return False
        self.state = "running"
        return True


def check_contract(run_type):
    run = run_type()
    assert run.cancel() == "cancelled"
    assert run.cancel() == "cancelled"
    assert not run.claim(), "worker claimed a cancelled run"
    running = run_type()
    assert running.claim()
    assert running.cancel() == "running"

if __name__ == "__main__":
    import difflib
    import inspect
    original = inspect.getsource(Run)
    mutant = original.replace('if self.state != "queued":', 'if self.state == "running":')
    if mutant == original:
        raise RuntimeError("Mutation did not land")
    print("".join(difflib.unified_diff(original.splitlines(True),mutant.splitlines(True),
                                     fromfile="baseline",tofile="mutant")),end="")
    namespace = {"dataclass": dataclass}
    exec(mutant,namespace)
    check_contract(Run)
    try:
        check_contract(namespace["Run"])
    except AssertionError as error:
        assert str(error) == "worker claimed a cancelled run"
        print("Baseline passed. Mutant caught:", error)
    else:
        raise AssertionError("Mutant escaped the contract check")
