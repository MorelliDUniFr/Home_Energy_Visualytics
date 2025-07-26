def format_value(value, unit):
    """
    Formats a numeric value with an appropriate SI prefix (k, M, G).

    Args:
        value (float): The numeric value to format.
        unit (str): The unit to append (e.g., 'W', 'Wh', 'B').

    Returns:
        str: A human-readable string like '1.23 kW' or '500.00 W'.
    """
    prefixes = ['', 'k', 'M', 'G']  # SI prefixes: none, kilo, mega, giga
    scale = 1000.0
    abs_value = abs(value)

    # Iterate from largest to smallest prefix
    for i in range(len(prefixes) - 1, -1, -1):
        threshold = scale ** i
        if abs_value >= threshold:
            scaled_value = value / threshold
            return f"{scaled_value:.2f} {prefixes[i]}{unit}"

    # For values smaller than 1.0 (or all thresholds), use base unit
    return f"{value:.2f} {prefixes[0]}{unit}"
