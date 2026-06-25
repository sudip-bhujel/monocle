from __future__ import annotations

from dataclasses import dataclass

from monocle.schema import Case


@dataclass(frozen=True)
class SandboxPolicy:
    allow_network: bool = False
    timeout_s: float = 2.0
    memory_mb: int = 256
    rollback_filesystem: bool = True


class FixtureOracle:
    def __init__(self, policy: SandboxPolicy | None = None) -> None:
        self.policy = policy or SandboxPolicy()

    def label(self, case: Case) -> str:
        if self.policy.allow_network:
            raise RuntimeError("unsafe oracle execution requires network blocking")
        return case.label

    def validate_policy(self) -> bool:
        return (
            not self.policy.allow_network
            and self.policy.timeout_s > 0
            and self.policy.memory_mb > 0
            and self.policy.rollback_filesystem
        )


def validate_fixture_oracle_boundary(oracle: FixtureOracle) -> None:
    if not isinstance(oracle, FixtureOracle) or not oracle.validate_policy():
        raise RuntimeError(
            "v1 only permits fixture oracle labels; real unsafe execution is blocked"
        )


def unsafe_execution_blocked() -> None:
    raise RuntimeError(
        "real unsafe command or code execution requires a tested sandbox implementation"
    )
