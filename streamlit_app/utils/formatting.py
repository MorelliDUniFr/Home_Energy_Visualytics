def format_value(value, unit):
    prefixes = ['', 'k', 'M', 'G']

    scale = 1000.0
    abs_value = abs(value)

    for i in range(len(prefixes) - 1, -1, -1):
        threshold = scale ** i
        if abs_value >= threshold:
            scaled_value = value / threshold
            return f"{scaled_value:.2f} {prefixes[i]}{unit}"
    return f"{value:.2f} {prefixes[0]}{unit}"  # for very small values
