"""Size units. Records store bytes; tables show MB with 1 MB = 2**20 bytes
(the convention of OMG and HAC++), labelled "MB (2^20 bytes)"."""

MiB = 2**20


def bits_per_gaussian(size_bytes: int, n_gaussians: int) -> float:
    """Storage bits per rendered (world-space) Gaussian."""
    return size_bytes * 8 / n_gaussians
