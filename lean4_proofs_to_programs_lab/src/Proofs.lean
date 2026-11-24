import Lean

namespace Proofs

-- A proposition is a type, and a proof is a term inhabiting it.
theorem identity (p : Prop) : p → p := by
  intro hp
  exact hp

-- We can prove a useful equation without writing any new axiom.
theorem add_zero (n : Nat) : n + 0 = n := by
  rfl

-- Reuse a theorem from the standard library.
theorem add_comm (a b : Nat) : a + b = b + a := by
  exact Nat.add_comm a b

-- A tiny proven property of an actual program.
def double (n : Nat) : Nat := n + n

theorem double_zero : double 0 = 0 := by
  rfl

theorem double_one : double 1 = 2 := by
  rfl

end Proofs
