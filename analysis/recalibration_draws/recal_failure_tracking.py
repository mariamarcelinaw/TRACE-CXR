"""No silent exclusion of fitting failures."""
FAILURES=('SINGLE_CLASS_ADAPTATION','NONCONVERGENCE','PERFECT_SEPARATION','NUMERICAL_OVERFLOW','INVALID_PROBABILITY','NEGATIVE_SLOPE','NEAR_ZERO_SLOPE','UNSTABLE_COEFFICIENT','THRESHOLD_UNDEFINED','EVALUATION_SINGLE_CLASS','MISSING_INPUT','OTHER_ERROR')
def status_for_error(e):
    s=str(e)
    return next((k for k in FAILURES if k in s),'OTHER_ERROR')
def rate(events,attempts):
    if attempts<=0:raise ValueError('attempt count must be positive')
    return len(events)/attempts
