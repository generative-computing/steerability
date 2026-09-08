"""Heuristic layer selection strategies."""


def late_third(num_layers: int) -> list[int]:
    """Return layer ids for the last third of the model.

    This is the default heuristic for behavior layer selection when the
    user does not specify explicit layer ids.

    Args:
        num_layers: Total number of layers in the model.

    Returns:
        List of layer ids.
    """
    start = num_layers - (num_layers // 3)
    return list(range(start, num_layers))
