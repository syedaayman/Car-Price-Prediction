"""utils.py — Shared helper functions."""

import pickle
import os

MIN_PRICE = 30000   # Floor price: minimum observed in training data


def load_pickle_file(file_path):
    """Load any pickled Python object from disk."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"File not found: {file_path}\n"
            "Run the Feature Engineering and Model Training notebooks first."
        )
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def get_artifacts_directory():
    """Return the absolute path to the project's artifacts/ folder."""
    src_dir      = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(src_dir)
    return os.path.join(project_root, 'artifacts')


def format_price_inr(price):
    """
    Format a numeric price into an Indian Rupee string.

    Linear Regression can predict negative values for very old / high-km
    cars (it extrapolates beyond the training distribution).  We floor at
    MIN_PRICE (the lowest observed price in the training data).

    Examples
    --------
    >>> format_price_inr(350000)  -> '₹3,50,000'
    >>> format_price_inr(-5000)   -> '₹30,000'  (floored at MIN_PRICE)
    """
    price = max(MIN_PRICE, int(round(price)))

    price_str  = str(price)
    if len(price_str) <= 3:
        return f"₹{price_str}"

    last_three = price_str[-3:]
    remaining  = price_str[:-3]

    groups = []
    while len(remaining) > 2:
        groups.append(remaining[-2:])
        remaining = remaining[:-2]
    if remaining:
        groups.append(remaining)

    groups.reverse()
    return f"₹{','.join(groups)},{last_three}"
