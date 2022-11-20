def round_to_nearest_n(data, rounding_level):
    """Rounds the input to the nearest value as specified by the rounding level."""

    return rounding_level * round(data / rounding_level)

