"""The settings renamed in 1.2 still work under their old names, with one warning."""
import warnings

from adloc.utils import translate_old_config


def test_old_names_are_translated_with_one_warning():
    config = {"min_picks": 5, "min_picks_ratio": 0.5, "min_score": 0.6}
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        translate_old_config(config)
    assert config == {"min_picks": 5, "ransac_sample_ratio": 0.5, "ransac_min_r2": 0.6, "min_r2": 0.6}
    assert len(w) == 1 and "min_score -> ransac_min_r2 and min_r2" in str(w[0].message)


def test_new_name_wins_and_new_configs_are_silent():
    config = {"min_score": 0.1, "ransac_min_r2": 0.6, "min_r2": 0.7}
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        assert translate_old_config(config) == {"ransac_min_r2": 0.6, "min_r2": 0.7}
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        translate_old_config({"ransac_min_r2": 0.6, "ransac_sample_ratio": 0.5})
    assert len(w) == 0
