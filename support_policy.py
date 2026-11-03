"""Fixed, transparent measurement-sufficiency policy; NOT a clinical detector.

It restricts confident 'suspected' outputs when measurements contradict the
claimed local change. Weak cases stay rejected/uncertain and are never hidden.
"""
import statistics
POLICY={'min_line_length_ratio':2.,'min_rms_ratio':1.5,'min_changed_leads':2,
        'min_peak_fraction':.20,'max_entropy':.85,'min_consecutive_support_sec':15.,
        'min_support_fraction':.20,'minimum_model_interval_sec':10.}
def support_summary(e):
    a,b=e['proposed_interval'];rows=[r for r in e['time_series'] if r[0]<b and r[1]>a]
    flags=[r[2]>=POLICY['min_line_length_ratio'] and r[3]>=POLICY['min_rms_ratio'] and r[7]>=POLICY['min_changed_leads'] and r[6]>=POLICY['min_peak_fraction'] and r[5]<=POLICY['max_entropy'] for r in rows]
    longest=run=total=0.;previous_end=None;support_intervals=[]
    for r,ok in zip(rows,flags):
        if ok:
            run=(run if previous_end is not None and abs(previous_end-r[0])<.01 else 0)+r[1]-r[0]
            total+=r[1]-r[0];longest=max(longest,run);support_intervals.append([r[0],r[1]])
        else:run=0
        previous_end=r[1]
    fraction=total/max(b-a,1.)
    passed=longest>=POLICY['min_consecutive_support_sec'] and fraction>=POLICY['min_support_fraction']
    return {'policy':POLICY,'meets_minimum_measurement_support':passed,'longest_support_run_sec':longest,'total_support_sec':total,'support_fraction':round(fraction,4),
      'support_bins':support_intervals,'allowed_decisions':['suspected','rejected','uncertain'] if passed else ['rejected','uncertain'],
      'meaning':'A fixed engineering evidence-sufficiency check, not seizure truth. Passing does not establish a seizure; failure can occur in real focal/low-amplitude events. Rejected and uncertain candidates remain visible.'}
