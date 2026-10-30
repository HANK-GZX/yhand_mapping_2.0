"""Reassemble frozen per-group validation heads without borrowing final weights."""
from copy import deepcopy
import json
from pathlib import Path
from thumb_monotonic_training import replace_thumb_flex_ext
from thumb_side_training import replace_thumb_side
from thumb_opposition_training import replace_thumb_opposition


def group_validation_reports(source,reference_report):
    reports=deepcopy(source.get('group_validation_reports',{}))
    if 'flex_ext'not in reports:
        reports['flex_ext']=str(source['thumb_side_training'].get('flex_ext_validation_report',source['thumb_side_training']['reference_report']))if source['schema_version']in(13,14)else str(reference_report)
    if source['schema_version']in(13,14):reports.setdefault('side',str(reference_report))
    return reports


def load_thumb_fold(source,reference_report,scheme,label):
    def read(report):return json.loads((Path(report)/'fold_models'/scheme/f'{label}.json').read_text())
    result=deepcopy(source);result['network']=read(source['thumb_flex_ext_training']['reference_report'])
    reports=group_validation_reports(source,reference_report)
    result=replace_thumb_flex_ext(result,read(reports['flex_ext']))
    if 'coordination' in reports:
        from thumb_coordination_training import replace_coordination
        return replace_coordination(result, read(reports['coordination']))
    if 'side'in reports:result=replace_thumb_side(result,read(reports['side']))
    if 'opposition'in reports:result=replace_thumb_opposition(result,read(reports['opposition']))
    return result
