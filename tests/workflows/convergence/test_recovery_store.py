"""Storage contract and concurrent reservation tests."""

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

from nsdw.workflows.convergence.recovery_reservations import (
    ReservationConflict,
    ReservationStore,
)
from nsdw.workflows.convergence.recovery_state import RecoveryRecord
from nsdw.workflows.convergence.recovery_store import (
    RecoveryReservationStore,
)


def test_sqlite_implements_storage_contract(tmp_path):
    store = ReservationStore(tmp_path / "recovery.sqlite")
    assert isinstance(store, RecoveryReservationStore)


def test_simultaneous_reservations_have_one_winner(tmp_path):
    path = tmp_path / "recovery.sqlite"
    ReservationStore(path)

    barrier = Barrier(8)

    def compete(index):
        store = ReservationStore(path)
        record = RecoveryRecord(
            scientific_identity="shared-scientific-recovery",
            execution_identity=f"attempt-{index}",
        )
        barrier.wait(timeout=10)

        try:
            store.reserve(record)
            return "reserved"
        except ReservationConflict:
            return "conflict"

    with ThreadPoolExecutor(max_workers=8) as executor:
        outcomes = list(executor.map(compete, range(8)))

    assert outcomes.count("reserved") == 1
    assert outcomes.count("conflict") == 7

    stored = ReservationStore(path).get(
        "shared-scientific-recovery"
    )
    assert stored is not None
    assert stored.execution_identity.startswith("attempt-")


def test_storage_contract_preserves_recovery_identity(tmp_path):
    store = ReservationStore(tmp_path / "recovery.sqlite")

    original = RecoveryRecord(
        scientific_identity="scientific-identity",
        execution_identity="lumi-execution",
    )

    reserved = store.reserve(original)
    restored = store.get(original.scientific_identity)

    assert restored == reserved
