"""Tests for the gradient metrics logged by CodebookModel (urgency, disagreement)."""

import torch

from scion.ops import (
    compute_instance_urgency,
    compute_gradient_disagreement,
)


# ---------------------------------------------------------------------------
# compute_instance_urgency
# ---------------------------------------------------------------------------


class TestComputeInstanceUrgency:
    def test_basic(self):
        pos_grad = torch.tensor([[1.0, 0.0, 0.0], [0.0, 3.0, 4.0]])
        urgency = compute_instance_urgency(pos_grad)
        assert urgency.shape == (2,)
        assert torch.isclose(urgency[0], torch.tensor(1.0))
        assert torch.isclose(urgency[1], torch.tensor(5.0))

    def test_with_scale_grad(self):
        pos_grad = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        scale_grad = torch.tensor([[2.0], [0.5]])
        urgency = compute_instance_urgency(pos_grad, scale_grad, scale_weight=1.0)
        assert urgency[0] > urgency[1]  # both position and scale contribute
        assert torch.isclose(urgency[0], torch.tensor(3.0))  # 1.0 + 1.0*2.0
        assert torch.isclose(urgency[1], torch.tensor(0.5))  # 0.0 + 1.0*0.5

    def test_zero_gradients(self):
        pos_grad = torch.zeros(5, 3)
        urgency = compute_instance_urgency(pos_grad)
        assert (urgency == 0).all()


# ---------------------------------------------------------------------------
# compute_gradient_disagreement
# ---------------------------------------------------------------------------


class TestComputeGradientDisagreement:
    def test_aligned_gradients(self):
        """All instances in same candidate push in same direction -> low disagreement."""
        pos_grad = torch.tensor([
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
        ])
        assignments = torch.tensor([0, 0, 0])
        scores = compute_gradient_disagreement(pos_grad, assignments, K=1)
        assert scores.shape == (1,)
        assert scores[0] < 0.01  # near-zero disagreement

    def test_opposing_gradients(self):
        """Instances push in opposite directions -> high disagreement."""
        pos_grad = torch.tensor([
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
        ])
        assignments = torch.tensor([0, 0])
        scores = compute_gradient_disagreement(pos_grad, assignments, K=1)
        assert scores[0] > 0.9  # near-maximum disagreement

    def test_multiple_candidates(self):
        """Each candidate scored independently."""
        pos_grad = torch.tensor([
            [1.0, 0.0, 0.0],   # candidate 0
            [1.0, 0.0, 0.0],   # candidate 0 (aligned)
            [1.0, 0.0, 0.0],   # candidate 1
            [-1.0, 0.0, 0.0],  # candidate 1 (opposing)
        ])
        assignments = torch.tensor([0, 0, 1, 1])
        scores = compute_gradient_disagreement(pos_grad, assignments, K=2)
        assert scores[0] < 0.01  # candidate 0: aligned
        assert scores[1] > 0.9   # candidate 1: opposing

    def test_single_instance_candidate(self):
        """Candidates with < 2 instances get score 0."""
        pos_grad = torch.tensor([[1.0, 0.0, 0.0]])
        assignments = torch.tensor([0])
        scores = compute_gradient_disagreement(pos_grad, assignments, K=2)
        assert scores[0] == 0.0  # only 1 instance
        assert scores[1] == 0.0  # no instances
