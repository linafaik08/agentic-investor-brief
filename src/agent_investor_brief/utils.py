def safe_divide(numerator, denominator, default=0):
    """Safely divide two numbers, returning default if denominator is zero"""
    try:
        if denominator == 0 or denominator is None:
            return default
        return numerator / denominator
    except (TypeError, ZeroDivisionError):
        return default

def safe_percentage_change(new_value, old_value, default=0):
    """Safely calculate percentage change"""
    if old_value == 0 or old_value is None:
        return default
    try:
        return ((new_value - old_value) / old_value) * 100
    except (TypeError, ZeroDivisionError):
        return default