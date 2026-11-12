"""Tests for the log parser and guards in agg_runs.py (stdlib only).

Run: python scripts/test_agg_runs.py
Three review cycles found parsing defects here that no test would have missed, so
these pin the shapes that actually occur in the campaign's logs, not hypotheticals.
"""
import json
import os
import tempfile
import unittest

import agg_runs


def write_log(text):
    d = tempfile.mkdtemp()
    run = os.path.join(d, "run")
    os.makedirs(os.path.join(run, "checkpoint_p0"))
    with open(os.path.join(run, "sf_log.txt"), "w") as f:
        f.write(text)
    return run


def launch(ts, experiment="run"):
    return "[%s][1] Saving configuration to train_dir/%s/config.json...\n" % (ts, experiment)


def collected(ts, frames, fps=1000.0):
    return "[%s][1] Collected {0: %d}, FPS: %.1f\n" % (ts, frames, fps)


def reward(ts, value):
    # The real line quotes the value: Avg episode reward: [(0, '13.6')]
    return "[%s][1] Avg episode reward: [(0, '%.3f')]\n" % (ts, value)


class SessionTest(unittest.TestCase):
    def test_single_session_wall_is_launch_to_last_line(self):
        run = write_log(
            launch("2026-09-02 19:17:45,094")
            + reward("2026-09-02 19:18:00,000", 1.0)
            + collected("2026-09-02 22:17:45,094", 500)
        )
        _, frames, _, fps, wall, sessions = agg_runs.parse_log(
            os.path.join(run, "sf_log.txt"))
        self.assertEqual(sessions, 1)
        self.assertEqual(frames, 500)
        self.assertEqual(fps, 1000.0)
        self.assertAlmostEqual(wall, 3.0, places=2)

    def test_relaunch_is_counted_from_its_marker_not_from_a_count_regression(self):
        # The second launch's first Collected *repeats* the previous value, which is
        # the shape m3_trial_4 already contains: a regression test would miss it.
        run = write_log(
            launch("2026-08-04 15:52:06,111")
            + collected("2026-08-04 17:14:31,529", 500)
            + launch("2026-08-04 17:14:32,000")
            + collected("2026-08-04 19:22:18,104", 500)
        )
        _, frames, _, _, wall, sessions = agg_runs.parse_log(
            os.path.join(run, "sf_log.txt"))
        self.assertEqual(sessions, 2)
        self.assertEqual(frames, 500)
        # 1.37 h + 2.13 h, not the 3.30 h of a first-to-last subtraction.
        self.assertAlmostEqual(wall, 3.50, places=1)

    def test_frames_reached_is_the_largest_not_the_last(self):
        run = write_log(
            launch("2026-07-31 22:31:03,259")
            + collected("2026-08-01 02:54:54,000", 50020352)
            + launch("2026-08-01 02:55:05,053")
            + collected("2026-08-01 03:10:00,000", 3932160)
        )
        _, frames, _, _, wall, sessions = agg_runs.parse_log(
            os.path.join(run, "sf_log.txt"))
        self.assertEqual(frames, 50020352)
        self.assertEqual(sessions, 2)
        self.assertGreater(wall, 0)

    def test_fps_follows_the_sample_behind_frames_reached(self):
        run = write_log(
            launch("2026-07-31 22:31:03,259")
            + collected("2026-08-01 02:54:54,000", 50020352, 9949.4)
            + launch("2026-08-01 02:55:05,053")
            + collected("2026-08-01 03:10:00,000", 3932160, 0.0)
        )
        _, _, _, fps, _, _ = agg_runs.parse_log(os.path.join(run, "sf_log.txt"))
        self.assertEqual(fps, 9949.4)


class CollectTest(unittest.TestCase):
    """collect() is what the tables are built from: the reward guard lives there."""

    def _row(self, text, name="run"):
        d = tempfile.mkdtemp()
        run = os.path.join(d, name)
        os.makedirs(run)
        with open(os.path.join(run, "config.json"), "w") as f:
            json.dump({"train_for_env_steps": 1000, "rnn_type": "gru",
                       "rnn_size": 512, "seed": None}, f)
        with open(os.path.join(run, "sf_log.txt"), "w") as f:
            f.write(text)
        return agg_runs.collect(d)[0]

    def test_reward_series_is_parsed_at_all(self):
        row = self._row(launch("2026-09-02 19:00:00,000")
                        + reward("2026-09-02 19:00:05,000", 1.0)
                        + reward("2026-09-02 19:00:10,000", 3.0))
        self.assertEqual(row["points"], 2)
        self.assertEqual(row["last20"], 2.0)

    def test_two_markers_publish_no_reward_statistic(self):
        row = self._row(launch("2026-09-02 19:00:00,000")
                        + reward("2026-09-02 19:00:05,000", 1.0)
                        + collected("2026-09-02 19:10:00,000", 900)
                        + launch("2026-09-02 20:00:00,000")
                        + reward("2026-09-02 20:00:05,000", 7.0)
                        + collected("2026-09-02 20:10:00,000", 900))
        self.assertEqual(row["sessions"], 2)
        self.assertEqual(row["group"], "restarted")
        self.assertEqual(row["last20"], "")
        self.assertEqual(row["final"], "")
        self.assertEqual(row["best"], "")
        self.assertEqual(row["fps"], "")
        # 10 min + 10 min, and deliberately *not* the 1.17 h a first-to-last
        # subtraction would give (19:00 -> 20:10).
        self.assertAlmostEqual(row["wall_h"], 0.33, places=2)


class GuardTest(unittest.TestCase):
    def test_blank_budget_is_a_named_problem_not_a_crash(self):
        self.assertFalse(agg_runs.is_finished(500, ""))
        self.assertFalse(agg_runs.is_finished(500, None))
        self.assertTrue(agg_runs.is_finished(500, 500))

    def test_validate_flags_headline_run_with_two_sessions(self):
        rows = [dict(run="gru_250m_baseline", group="reported", frames_reached=250019840,
                     budget=250000000, sessions=2)]
        problems = agg_runs.validate(rows, tempfile.mkdtemp())
        self.assertTrue(any("holds 2 sessions" in p for p in problems), problems)

    def test_validate_flags_headline_run_that_stopped_early(self):
        rows = [dict(run="gru_250m_baseline", group="reported", frames_reached=13430000,
                     budget=250000000, sessions=1)]
        problems = agg_runs.validate(rows, tempfile.mkdtemp())
        self.assertTrue(any("headline row must have finished" in p for p in problems),
                        problems)

    def test_validate_flags_unreadable_budget_by_name(self):
        rows = [dict(run="gdn_50m_seed1", group="reported", frames_reached=50020352,
                     budget="", sessions=1)]
        problems = agg_runs.validate(rows, tempfile.mkdtemp())
        self.assertTrue(any("gdn_50m_seed1 has no readable" in p for p in problems),
                        problems)


class SignatureTest(unittest.TestCase):
    def test_engine_generated_ini_does_not_change_the_code_signature(self):
        a = ("diff --git a/_vizdoom.ini b/_vizdoom.ini\nindex 1..2\n--- a/_vizdoom.ini\n"
             "+++ b/_vizdoom.ini\n-# generated Fri Jun  5 21:47:07 2026\n"
             "+# generated Wed Sep  2 01:11:06 2026\n"
             "diff --git a/models/train.py b/models/train.py\nindex 3..4\n+x = 1\n")
        b = a.replace("Fri Jun  5 21:47:07 2026", "Mon Jan  1 00:00:00 2026").replace(
            "Wed Sep  2 01:11:06 2026", "Tue Jan  2 00:00:00 2026")
        self.assertEqual(agg_runs.code_signature(a), agg_runs.code_signature(b))

    def test_a_code_change_does_change_the_signature(self):
        a = "diff --git a/models/train.py b/models/train.py\nindex 3..4\n+x = 1\n"
        self.assertNotEqual(agg_runs.code_signature(a),
                            agg_runs.code_signature(a.replace("x = 1", "x = 2")))


if __name__ == "__main__":
    unittest.main(verbosity=2)
