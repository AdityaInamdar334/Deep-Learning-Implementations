import Proofs

def main : IO Unit := do
  IO.println "Lean 4: proof checking and programming in one language"
  IO.println s!"double 21 = {Proofs.double 21}"
