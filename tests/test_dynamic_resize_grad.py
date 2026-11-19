"""Tests for PyTorch autograd safety during dynamic parameter resizing.

Failure mode: an IndexBackward0 crash after an instance resize.

    PyTorch creates ONE AccumulateGrad node per leaf parameter, shared across
    ALL forward passes that use that parameter.  When a training loop holds
    `loss` and `loss_dict` from step T-1 alive (for logging), those tensors
    keep a reference chain alive:

        loss_prev.grad_fn → ... → IndexBackward0_T1 → AccumulateGrad

    After an instance resize replaces `instance_opacities.data` (via
    torch.cat), the same AccumulateGrad is now also
    referenced by step T's forward:

        loss.grad_fn → ... → IndexBackward0_T → AccumulateGrad

    During loss.backward(), PyTorch processes AccumulateGrad with ref_count=2,
    which triggers BOTH IndexBackward0_T AND IndexBackward0_T1.  But
    IndexBackward0_T1 was recorded with the old [N_old, 1] shape, while the
    parameter now has shape [N_new, 1], causing:

        RuntimeError: Function IndexBackward0 returned an invalid gradient at
        index 0 - got [N_new, 1] but expected shape compatible with [N_old, 1]

The remedy (gsplat-style Parameter replacement):
    After resize, create entirely NEW nn.Parameter objects for the resized
    instance params.  Each new Parameter gets a fresh AccumulateGrad node.
    Stale backward graph edges from previous steps point to the OLD (now
    orphaned) AccumulateGrad and can never conflict with the current step.

    This follows gsplat's _update_param_with_optimizer pattern, also used
    by 3DGS-MCMC.  See CodebookMixin.replace_instance_params(), called from
    scion.engine.optim.rebind_params().
"""

import pytest
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def simulate_forward(param: nn.Parameter, n_rows: int) -> dict:
    """Forward pass that indexes `param` with n_rows indices — creates IndexBackward0."""
    idx = torch.arange(n_rows, device=param.device)
    selected = param[idx]
    return {"main_loss": selected.mean()}


def find_accumulate_grad(node):
    """Walk a grad_fn DAG and return the AccumulateGrad node, or None."""
    if node is None:
        return None
    if type(node).__name__ == "AccumulateGrad":
        return node
    for fn, _ in (node.next_functions or []):
        result = find_accumulate_grad(fn)
        if result is not None:
            return result
    return None


# ---------------------------------------------------------------------------
# AccumulateGrad sharing
# ---------------------------------------------------------------------------

class TestAccumulateGradSharing:
    """Confirm PyTorch shares ONE AccumulateGrad per leaf parameter."""

    def test_shared_across_two_forwards_same_size(self):
        """AccumulateGrad is the same object for two identical-sized forwards."""
        param = nn.Parameter(torch.zeros(100, 1))

        loss1 = param[torch.arange(50)].mean()
        loss2 = param[torch.arange(80)].mean()

        accum1 = find_accumulate_grad(loss1.grad_fn)
        accum2 = find_accumulate_grad(loss2.grad_fn)

        assert accum1 is not None, "AccumulateGrad not found in loss1 graph"
        assert accum2 is not None, "AccumulateGrad not found in loss2 graph"
        assert accum1 is accum2, (
            "AccumulateGrad must be the same Python object across forward passes "
            "for the same parameter — this is the sharing that causes the crash."
        )

    def test_shared_across_two_forwards_after_resize(self):
        """AccumulateGrad stays the same object even after param.data is replaced."""
        param = nn.Parameter(torch.zeros(100, 1))

        loss1 = param[torch.arange(100)].mean()
        accum_before = find_accumulate_grad(loss1.grad_fn)

        # Resize the parameter's storage
        param.data = torch.cat([param.data, torch.zeros(20, 1)])

        loss2 = param[torch.arange(120)].mean()
        accum_after = find_accumulate_grad(loss2.grad_fn)

        assert accum_before is accum_after, (
            "AccumulateGrad identity must survive param.data replacement — "
            "it is tied to the nn.Parameter Python object, not its storage."
        )


# ---------------------------------------------------------------------------
# Single-step resize (safe baseline)
# ---------------------------------------------------------------------------

class TestSingleStepResize:
    """Resize BEFORE the only forward pass is always safe."""

    def test_resize_before_forward_no_crash(self):
        """If no prior backward graph is alive, resize + forward + backward is fine."""
        param = nn.Parameter(torch.zeros(100, 1))
        optimizer = torch.optim.Adam([param], lr=1e-3)

        # Step T-1: complete cycle with explicit cleanup
        loss_dict_prev = simulate_forward(param, 100)
        loss_prev = sum(loss_dict_prev.values())
        loss_prev.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        del loss_dict_prev, loss_prev  # release backward graph

        # Resize — no stale graph, so safe
        param.data = torch.cat([param.data, torch.zeros(20, 1)])
        param.grad = None

        loss_dict = simulate_forward(param, 120)
        loss = sum(loss_dict.values())
        loss.backward()  # must not raise


# ---------------------------------------------------------------------------
# Cross-step crash
# ---------------------------------------------------------------------------

class TestCrossStepCrash:
    """Reproduce and fix the IndexBackward0 crash from stale loss references."""

    def test_stale_loss_dict_causes_crash(self):
        """Step T-1 loss_dict alive + resize + step T backward → IndexBackward0 crash.

        A training loop keeps `loss` and `loss_dict` from the previous step
        alive for logging, and a BEFORE callback resizes the parameter between
        steps.
        """
        param = nn.Parameter(torch.zeros(100, 1))
        optimizer = torch.optim.Adam([param], lr=1e-3)

        # Step T-1: forward + backward — intentionally keep loss alive
        loss_dict_prev = simulate_forward(param, 100)
        loss_prev = sum(loss_dict_prev.values())
        loss_prev.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        # NOTE: loss_dict_prev and loss_prev are NOT deleted — the loop holds them

        # BEFORE callback for step T: resize (stale graph still alive)
        param.data = torch.cat([param.data, torch.zeros(20, 1)])
        param.grad = None

        # Step T: forward + backward
        loss_dict = simulate_forward(param, 120)
        loss = sum(loss_dict.values())

        with pytest.raises(RuntimeError, match="IndexBackward0|invalid gradient"):
            loss.backward()

    def test_fix_detach_inplace_on_combined_loss(self):
        """Fix: detach_() the combined loss tensor before resize.

        If the model returns {"loss": combined}, then
        `loss = functools.reduce(torch.add, loss_dict.values())` is `combined`
        (same Python object, since reduce of one element is identity).

        Storing `prev_combined_loss = combined` and calling
        `prev_combined_loss.detach_()` in the BEFORE callback severs the
        grad_fn chain in-place — affecting all Python aliases (including the
        loop's `loss` and `loss_dict["loss"]` variables) because they all
        point to the same tensor object.
        """
        param = nn.Parameter(torch.zeros(100, 1))
        optimizer = torch.optim.Adam([param], lr=1e-3)

        # Step T-1: forward + backward
        loss_dict_prev = simulate_forward(param, 100)
        # Simulate a loss function returning {"loss": combined}
        combined_prev = sum(loss_dict_prev.values())
        loss_dict_prev = {"loss": combined_prev}   # single-entry dict
        loss_prev = combined_prev                  # trainer: loss = reduce(add, dict.values())
        prev_combined_loss_ref = combined_prev     # model stores this reference

        loss_prev.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        # The loop still holds loss_dict_prev and loss_prev

        # THE FIX: BEFORE callback calls prev_combined_loss.detach_() first
        prev_combined_loss_ref.detach_()
        # All aliases (loss_prev, loss_dict_prev["loss"], prev_combined_loss_ref)
        # now point to a detached leaf — no grad_fn chain.

        # BEFORE callback: now safe to resize
        param.data = torch.cat([param.data, torch.zeros(20, 1)])
        param.grad = None

        # Step T: forward + backward — no crash
        loss_dict = simulate_forward(param, 120)
        loss = sum(loss_dict.values())
        loss.backward()

    def test_fix_del_stale_references(self):
        """Alternative fix: explicitly delete all stale graph references.

        Works, but requires knowing every variable holding the graph.
        detach_() is preferred since it works regardless of how many variables
        the framework holds internally.
        """
        param = nn.Parameter(torch.zeros(100, 1))
        optimizer = torch.optim.Adam([param], lr=1e-3)

        loss_dict_prev = simulate_forward(param, 100)
        loss_prev = sum(loss_dict_prev.values())
        loss_prev.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        # Delete ALL Python references to step T-1's backward graph
        del loss_dict_prev, loss_prev

        param.data = torch.cat([param.data, torch.zeros(20, 1)])
        param.grad = None

        loss_dict = simulate_forward(param, 120)
        loss = sum(loss_dict.values())
        loss.backward()  # OK


# ---------------------------------------------------------------------------
# detach_() in-place semantics
# ---------------------------------------------------------------------------

class TestDetachInPlaceSemantics:
    """Verify detach_() mutates the tensor object in-place across all aliases."""

    def test_detach_inplace_affects_all_aliases(self):
        """All Python names pointing at the same tensor object see the detach."""
        param = nn.Parameter(torch.zeros(50, 1))
        idx = torch.arange(50)
        loss = param[idx].mean()

        alias_a = loss   # same Python object
        alias_b = loss   # same Python object

        assert alias_a.grad_fn is not None
        assert alias_b.grad_fn is not None

        alias_a.detach_()  # in-place on the shared object

        assert alias_a.grad_fn is None
        assert alias_b.grad_fn is None  # affected — same object
        assert loss.grad_fn is None     # affected — same object

    def test_detach_copy_does_not_affect_original(self):
        """detach() (non-inplace) creates a new tensor; original keeps its grad_fn."""
        param = nn.Parameter(torch.zeros(50, 1))
        idx = torch.arange(50)
        loss = param[idx].mean()

        detached_copy = loss.detach()   # new tensor, no grad_fn
        assert loss.grad_fn is not None  # original unchanged
        assert detached_copy.grad_fn is None

    def test_reduce_single_element_is_identity(self):
        """functools.reduce(torch.add, [x]) returns x, not a copy.

        This is why returning {"loss": combined} from the loss function
        ensures that a loop's `loss = reduce(add, loss_dict.values())`
        IS the same Python object as `combined`, so detach_() on our stored
        reference also breaks the trainer's `loss` variable.
        """
        import functools
        param = nn.Parameter(torch.zeros(10, 1))
        combined = param.sum()
        result = functools.reduce(torch.add, [combined])
        assert result is combined, "reduce of single element must return the same object"


# ---------------------------------------------------------------------------
# Multi-step training simulation (integration)
# ---------------------------------------------------------------------------

class TestMultiStepTraining:
    """End-to-end simulation of the training loop with the fix applied."""

    def test_three_steps_with_resize_each_step(self):
        """Three consecutive steps with resize at every BEFORE callback — no crash.

        After each resize, the Adam optimizer state (exp_avg, exp_avg_sq) has
        the old shape and must be cleared.
        """
        import functools

        param = nn.Parameter(torch.zeros(100, 1))
        optimizer = torch.optim.Adam([param], lr=1e-3)

        prev_combined_loss = None
        n = 100

        for step in range(3):
            # --- BEFORE callback ---
            if prev_combined_loss is not None:
                prev_combined_loss.detach_()  # THE FIX: break stale grad_fn chain

            # Instance resize (every step for stress testing)
            add = 10
            param.data = torch.cat([param.data, torch.zeros(add, 1)])
            param.grad = None
            n += add

            # Clear stale Adam state (shape mismatch in exp_avg/exp_avg_sq).
            optimizer.state.clear()

            # --- Forward pass ---
            loss_dict = simulate_forward(param, n)
            combined = sum(loss_dict.values())
            loss_dict = {"loss": combined}                  # single-entry dict
            loss = functools.reduce(torch.add, loss_dict.values())  # trainer
            prev_combined_loss = combined                   # model stores ref

            # --- Backward + optimizer step ---
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        # All steps completed without RuntimeError
        assert param.shape[0] == 130  # 100 + 10*3

    def test_stale_metrics_dict_causes_crash(self):
        """Stale metrics_dict with grad_fn tensors also keeps IndexBackward0 alive.

        Even after detach_() on combined loss, if metrics_dict from the
        previous step holds tensors computed from model_outputs (e.g. PSNR),
        those have grad_fn chains through IndexBackward0 → AccumulateGrad.
        This is the SECOND source of stale graph retention (the first being
        loss/loss_dict).

        Simulates: model.get_metrics_dict returns psnr = f(param[indices]),
        trainer holds metrics_dict across steps.
        """
        import functools

        param = nn.Parameter(torch.zeros(100, 1))
        optimizer = torch.optim.Adam([param], lr=1e-3)

        # Step T-1: forward + backward, keep stale metrics_dict
        idx = torch.arange(100)
        selected = param[idx]  # IndexBackward0
        loss_T1 = selected.mean()
        # Simulate a metric computed from model_outputs (like PSNR)
        metrics_dict_prev = {"psnr": (selected ** 2).mean()}  # has grad_fn!

        combined_T1 = loss_T1
        loss_dict_prev = {"loss": combined_T1}
        loss_prev = functools.reduce(torch.add, loss_dict_prev.values())
        prev_combined_ref = combined_T1

        loss_prev.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        # BEFORE callback: detach combined loss
        prev_combined_ref.detach_()

        # Resize
        param.data = torch.cat([param.data, torch.zeros(20, 1)])
        param.grad = None
        optimizer.state.clear()

        # Step T: forward + backward
        loss_dict = simulate_forward(param, 120)
        combined = sum(loss_dict.values())
        loss = functools.reduce(torch.add, [combined])

        # CRASH: metrics_dict_prev["psnr"] still holds stale IndexBackward0
        with pytest.raises(RuntimeError, match="IndexBackward0|invalid gradient"):
            loss.backward()



# ---------------------------------------------------------------------------
# New Parameter (gsplat-style — what scion does)
# ---------------------------------------------------------------------------

class TestNewParameterFix:
    """Verify that creating a new nn.Parameter breaks AccumulateGrad sharing.

    This is what CodebookMixin.replace_instance_params does, following
    gsplat's _update_param_with_optimizer pattern.
    """

    def test_new_parameter_gets_fresh_accumulate_grad(self):
        """A new nn.Parameter wrapping the same data has a DIFFERENT AccumulateGrad."""
        old_param = nn.Parameter(torch.zeros(100, 1))

        loss1 = old_param[torch.arange(100)].mean()
        accum_old = find_accumulate_grad(loss1.grad_fn)

        # Create new Parameter (same data, new Python object)
        new_param = nn.Parameter(old_param.data)

        loss2 = new_param[torch.arange(100)].mean()
        accum_new = find_accumulate_grad(loss2.grad_fn)

        assert accum_old is not None
        assert accum_new is not None
        assert accum_old is not accum_new, (
            "New nn.Parameter must get a DIFFERENT AccumulateGrad node — "
            "this is what makes the gsplat-style fix work."
        )

    def test_new_parameter_prevents_crash_with_stale_loss(self):
        """Stale loss from step T-1 + resize + new Parameter → no crash.

        The stale loss points to the OLD AccumulateGrad.  Step T uses the
        NEW Parameter (fresh AccumulateGrad), so backward never reaches
        the old one.  No IndexBackward0 shape conflict.
        """
        param = nn.Parameter(torch.zeros(100, 1))
        optimizer = torch.optim.Adam([param], lr=1e-3)

        # Step T-1: forward + backward — keep loss alive
        loss_dict_prev = simulate_forward(param, 100)
        loss_prev = sum(loss_dict_prev.values())
        loss_prev.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        # loss_dict_prev and loss_prev NOT deleted — the loop holds them

        # BEFORE callback: resize
        param.data = torch.cat([param.data, torch.zeros(20, 1)])

        # THE FIX: create new Parameter (gsplat pattern)
        new_param = nn.Parameter(param.data)

        # Update optimizer to use new param
        del optimizer.state[param]
        optimizer.param_groups[0]["params"] = [new_param]

        # Step T: forward + backward with NEW parameter — no crash
        loss_dict = simulate_forward(new_param, 120)
        loss = sum(loss_dict.values())
        loss.backward()  # must not raise

    def test_new_parameter_prevents_crash_with_stale_metrics(self):
        """Stale metrics_dict + stale loss + resize + new Parameter → no crash.

        Both sources of stale graph retention (loss and metrics) are harmless
        because they point to the OLD AccumulateGrad.  The new Parameter's
        fresh AccumulateGrad is disconnected from all stale references.
        """
        import functools

        param = nn.Parameter(torch.zeros(100, 1))
        optimizer = torch.optim.Adam([param], lr=1e-3)

        # Step T-1: forward with metrics (like PSNR from model_outputs)
        idx = torch.arange(100)
        selected = param[idx]  # IndexBackward0 → AccumulateGrad(param)
        loss_T1 = selected.mean()
        metrics_dict_prev = {"psnr": (selected ** 2).mean()}  # has grad_fn!

        combined_T1 = loss_T1
        loss_dict_prev = {"loss": combined_T1}
        loss_prev = functools.reduce(torch.add, loss_dict_prev.values())

        loss_prev.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        # The loop holds loss_prev, loss_dict_prev, AND metrics_dict_prev

        # BEFORE callback: resize
        param.data = torch.cat([param.data, torch.zeros(20, 1)])

        # THE FIX: create new Parameter (gsplat pattern)
        new_param = nn.Parameter(param.data)
        del optimizer.state[param]
        optimizer.param_groups[0]["params"] = [new_param]

        # Step T: forward + backward with NEW parameter — no crash
        loss_dict = simulate_forward(new_param, 120)
        loss = sum(loss_dict.values())
        loss.backward()  # must not raise — stale refs point to old AccumulateGrad

    def test_multi_step_with_parameter_replacement(self):
        """Three consecutive steps with resize + Parameter replacement each step.

        Simulates a full training loop:
        1. BEFORE callback: resize param.data, create new Parameter, update optimizer
        2. Forward pass: use new Parameter (creates IndexBackward0 → new AccumulateGrad)
        3. Backward + optimizer step
        4. Trainer holds loss/metrics alive for logging (stale references)
        """
        param = nn.Parameter(torch.zeros(100, 1))
        optimizer = torch.optim.Adam([param], lr=1e-3)
        n = 100

        for step in range(3):
            # --- BEFORE callback: resize + replace Parameter ---
            add = 10
            param.data = torch.cat([param.data, torch.zeros(add, 1)])
            n += add

            # Create new Parameter (gsplat pattern)
            new_param = nn.Parameter(param.data)
            if param in optimizer.state:
                del optimizer.state[param]
            optimizer.param_groups[0]["params"] = [new_param]
            param = new_param  # update local reference

            # --- Forward pass ---
            idx = torch.arange(n, device=param.device)
            selected = param[idx]
            loss_val = selected.mean()
            # Simulate stale metrics (with grad_fn, not detached)
            _metrics_dict = {"psnr": (selected ** 2).mean()}  # noqa: F841

            loss_dict = {"main_loss": loss_val}
            loss = sum(loss_dict.values())

            # --- Backward + optimizer step ---
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            # loss, loss_dict, _metrics_dict survive to next iteration

        assert param.shape[0] == 130  # 100 + 10*3
