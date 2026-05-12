# Canonical task -> condition name mappings for EV loading.
# Condition code 1 = first in list, 2 = second, etc.
# You can edit this file or regenerate with: python fm-fmri/build_ev_condition_mappings.py --ev_root .

from typing import Dict, List

TASK_CONDITION_MAP: Dict[str, List[str]] = {
    "emotion": ['fear', 'neut'],
    "gambling": ['loss', 'loss_event', 'neut_event', 'win', 'win_event'],
    "language": ['cue', 'math', 'present_math', 'present_story', 'question_math', 'question_story', 'response_math', 'response_story', 'story'],
    "motor": ['cue', 'lf', 'lh', 'rf', 'rh', 't'],
    "relational": ['error', 'match', 'relation'],
    "social": ['mental', 'mental_resp', 'other_resp', 'rnd'],
    "WM": ['0bk_body', '0bk_cor', '0bk_err', '0bk_faces', '0bk_nlr', '0bk_places', '0bk_tools', '2bk_body', '2bk_cor', '2bk_err', '2bk_faces', '2bk_nlr', '2bk_places', '2bk_tools', 'all_bk_cor', 'all_bk_err'],
}
