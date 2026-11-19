"""NaN-abort tolerance is opt-in, never exits 0, and never tolerates OOM.

The old handler substring-matched ("inf", "cuda error", "cudnn") on the
exception message and exited 0, so OOM'd or crashed runs were recorded as
completed. These tests pin the replacement classifier's behaviour.
"""

import torch

from scion.cli import NAN_ABORT_EXIT_CODE, _is_tolerable_nan_abort


class TestAbortClassifier:
    def test_non_finite_guard_message_is_tolerable(self):
        exc = RuntimeError("Non-finite values in world_means before rasterization")
        assert _is_tolerable_nan_abort(exc)

    def test_nan_loss_is_tolerable(self):
        assert _is_tolerable_nan_abort(RuntimeError("loss became NaN at step 1200"))

    def test_device_side_assert_is_tolerable(self):
        exc = RuntimeError("CUDA error: device-side assert triggered")
        assert _is_tolerable_nan_abort(exc)

    def test_oom_exception_type_is_never_tolerable(self):
        exc = torch.cuda.OutOfMemoryError(
            "CUDA out of memory. Tried to allocate 2.00 GiB"
        )
        assert not _is_tolerable_nan_abort(exc)

    def test_oom_message_is_never_tolerable(self):
        assert not _is_tolerable_nan_abort(RuntimeError("CUDA error: out of memory"))

    def test_old_substring_false_positives_are_gone(self):
        # The old matcher fired on any message containing "inf" or "cuda error".
        assert not _is_tolerable_nan_abort(
            RuntimeError("could not infer dtype of NoneType")
        )
        assert not _is_tolerable_nan_abort(
            RuntimeError("CUDA error: an illegal memory access was encountered")
        )

    def test_non_runtime_errors_are_not_tolerable(self):
        assert not _is_tolerable_nan_abort(ValueError("nan"))

    def test_exit_code_is_distinct_and_nonzero(self):
        assert NAN_ABORT_EXIT_CODE != 0
