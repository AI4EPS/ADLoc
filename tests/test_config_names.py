"""The settings renamed in 1.2 still work under their old names, with one warning."""
import warnings

from adloc.utils import translate_old_config


def test_old_names_are_translated_with_one_warning():
    config = {"min_picks": 5, "min_picks_ratio": 0.5, "min_p_picks": 1.0, "min_s_picks": 1.5, "min_score": 0.6}
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        translate_old_config(config)
    assert config == {"min_picks": 5, "ransac_sample_ratio": 0.5, "min_p_score_sum": 1.0, "min_s_score_sum": 1.5, "min_r2": 0.6}
    assert len(w) == 1 and "min_score -> min_r2" in str(w[0].message)


def test_new_name_wins_and_new_configs_are_silent():
    config = {"min_score": 0.1, "min_r2": 0.6}
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        assert translate_old_config(config) == {"min_r2": 0.6}
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        translate_old_config({"min_r2": 0.6, "ransac_sample_ratio": 0.5})
    assert len(w) == 0
