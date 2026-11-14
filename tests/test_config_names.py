"""A config from before 1.2 still runs: its names are translated, with one warning."""
import warnings

from adloc.utils import translate_old_config


def test_old_config_is_translated_with_one_warning():
    config = {"min_picks": 5, "min_picks_ratio": 0.5, "min_score": 0.6, "min_p_picks": 1.5, "min_s_picks": 1.5}
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        translate_old_config(config)
    assert config == {"min_picks": 5, "ransac_sample_ratio": 0.5, "ransac_min_r2": 0.6,
                      "ransac_min_p_score_sum": 1.5, "ransac_min_s_score_sum": 1.5}
    assert len(w) == 1 and "min_p_picks -> ransac_min_p_score_sum" in str(w[0].message)


def test_new_config_is_left_alone():
    config = {"min_picks": 5, "min_p_picks": 1, "min_s_picks": 1, "ransac_min_r2": 0.6}  # min_p_picks: a count, since 1.2
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        assert translate_old_config(dict(config)) == config
    assert len(w) == 0
