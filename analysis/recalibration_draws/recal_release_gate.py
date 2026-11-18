"""Pipeline variants that have not been independently verified remain fail-closed."""
def corrected_release(*args,**kwargs):
    raise PermissionError('NOT_EXECUTED_UPSTREAM_CORRECTED_PIPELINE_BLOCKED: manual scientific validation and adapter integration required')

def historical_extension_release(acknowledge_retrospective=False,source_version=''):
    if not acknowledge_retrospective or source_version!='KermanyNorm':raise PermissionError('retrospective-analysis acknowledgement and KermanyNorm version required')
    return 'RETROSPECTIVE_SENSITIVITY_ANALYSIS'
