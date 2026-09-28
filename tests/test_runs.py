"""Whole evening runs through HAL's runner, offline: replay fixtures plus a scripted model.

These are the replay suite from the plan. Real Home Assistant data (2026-09-22,
a sunny day before the solar fix) drives most of them; scenarios are derived
from it (zero output, dropout, cloudy) so the numbers stay realistic.
"""

from __future__ import annotations

import json
from datetime import date

from hal9000.coordinator.runner import RunOptions, open_database, run_evening
from hal9000.core import registry as tool_catalogue
from hal9000.solar.agents.solar import SOLAR_TAGS, build_solar_agent, solar_registry
from hal9000.solar.agents.triage import (
    FORBIDDEN_TRIAGE_TAGS,
    NOT_FOR_TRIAGE,
    triage_props,
    triage_registry,
)
from tests.fakes import FakeLLM, tool_call
from tests.fixture_builder import EXPORT, IMPORT, REAL_2026_09_22, SOLAR, fixture

DAY = date(2026, 9, 22)
D = DAY.isoformat()


def clear_day():
    return fixture(
        DAY, REAL_2026_09_22[SOLAR], REAL_2026_09_22[IMPORT], REAL_2026_09_22[EXPORT], 0.97
    )


def zero_day():
    imp = [v + REAL_2026_09_22[SOLAR][h] * 0.5 for h, v in enumerate(REAL_2026_09_22[IMPORT])]
    return fixture(DAY, [0.0] * 24, imp, [0.0] * 24, 0.97)


def dropout_day():
    solar = list(REAL_2026_09_22[SOLAR])
    export = list(REAL_2026_09_22[EXPORT])
    solar[12] = solar[13] = 0.02
    export[12] = export[13] = 0.0
    return fixture(DAY, solar, REAL_2026_09_22[IMPORT], export, 0.97)


def cloudy_day():
    solar = [v * 0.35 for v in REAL_2026_09_22[SOLAR]]
    export = [
        max(e - 0.6 * s * 2, 0.0) for s, e in zip(REAL_2026_09_22[SOLAR], REAL_2026_09_22[EXPORT])
    ]
    return fixture(DAY, solar, REAL_2026_09_22[IMPORT], export, 0.35)


ROUTINE = [
    tool_call("get_home_energy", date=D),
    tool_call("get_weather", date=D),
    tool_call("assess_performance"),
    tool_call("compute_savings"),
    tool_call("check_net_position"),
]


def run(write_fixture, fx, script, **kw):
    llm = FakeLLM(list(script))
    options = RunOptions(replay=write_fixture(fx), generate_response=llm, **kw)
    return run_evening(options), llm


# --- registries -------------------------------------------------------------------


def test_solar_registry_is_exactly_the_plan(registered_tools):
    names = {a.name for a in solar_registry().get_actions()}
    assert names == {
        "get_home_energy", "get_weather", "assess_performance", "compute_savings",
        "check_net_position", "get_hourly_anomaly_profile", "get_recent_history",
        "run_fault_triage", "check_alert_state", "record_alert", "send_telegram",
        "write_nightly_record", "deliver_report",
    }  # fmt: skip
    assert "terminate" not in names


def test_triage_registry_has_no_forbidden_tools(registered_tools):
    import hal9000.solar.tools  # noqa: F401

    for action in triage_registry().get_actions():
        tags = set(tool_catalogue.tools[action.name]["tags"])
        assert not tags & set(FORBIDDEN_TRIAGE_TAGS), action.name
    names = {a.name for a in triage_registry().get_actions()}
    assert (
        "submit_verdict" in names and "record_alert" not in names and "deliver_report" not in names
    )


def test_triage_props_drop_delivery_and_stores():
    props = {name: object() for name in NOT_FOR_TRIAGE} | {"run": object()}
    assert set(triage_props(props)) == {"run"}


def test_no_analysis_tool_takes_a_number_from_the_model(registered_tools):
    for action in solar_registry().get_actions():
        if action.name in ("get_recent_history",):
            continue  # a day count, not a reported figure
        for spec in action.parameters.get("properties", {}).values():
            assert spec.get("type") not in ("number", "integer", "array"), action.name


# --- replay runs ------------------------------------------------------------------


def test_clear_day_llm_report(hal_env, write_fixture):
    result, llm = run(
        write_fixture,
        clear_day(),
        ROUTINE
        + [
            tool_call("write_nightly_record"),
            tool_call("deliver_report", narrative="Production was normal today, worth $4.14."),
        ],
    )
    assert result.status == "ok"
    assert result.renderer == "llm"
    assert result.report_delivered and result.record_written and result.announced
    assert result.band == "normal"
    assert result.iterations == 7 <= 12
    assert set(llm.tiers) == {"fast"}
    # every prompt starts with an identical goals prefix (the cache prerequisite)
    firsts = {p.messages[0]["content"] for p in llm.calls}
    assert len(firsts) == 1
    assert all(p.metadata.get("cache_key") == "hal-solar" for p in llm.calls)
    transcript = json.loads(open(result.transcript_path).read())
    assert transcript["figures"]["production_kwh"] == "13.0"
    assert transcript["outbox"][-1]["channel"] == "sonos"


def test_prose_instead_of_tool_still_delivers(hal_env, write_fixture):
    """The model answers in prose on step 2: HAL's completeness check covers it."""
    script = [tool_call("get_home_energy", date=D), "Everything looks fine today!"] + ["Done."] * 12
    result, llm = run(write_fixture, clear_day(), script)
    assert len(llm.calls) == 3  # two prose replies in a row end the loop
    assert result.report_delivered and result.record_written
    assert result.renderer == "fallback"
    assert result.status == "degraded"
    assert "13.0" in result.report_text


def test_guard_two_strikes_falls_back(hal_env, write_fixture):
    script = ROUTINE + [
        tool_call("write_nightly_record"),
        tool_call("deliver_report", narrative="The panels made about 12.5 kilowatt-hours."),
        tool_call("deliver_report", narrative="The panels made roughly 12.9 kilowatt-hours."),
    ]
    result, _ = run(write_fixture, clear_day(), script)
    assert result.renderer == "fallback"
    assert any("twice" in r for r in result.degraded_reasons)
    assert result.report_text.startswith("Solar report for Tuesday")


def test_guard_rejection_then_correction(hal_env, write_fixture):
    script = ROUTINE + [
        tool_call("write_nightly_record"),
        tool_call("deliver_report", narrative="About 12.5 kilowatt-hours today."),
        tool_call("deliver_report", narrative="The panels made 13.0 kilowatt-hours today."),
    ]
    result, _ = run(write_fixture, clear_day(), script)
    assert result.renderer == "llm" and result.status == "ok"


def test_incomplete_report_is_sent_back(hal_env, write_fixture):
    script = (
        [
            tool_call("get_home_energy", date=D),
            tool_call("deliver_report", narrative="Production was fine."),
        ]
        + ROUTINE[1:]
        + [tool_call("deliver_report", narrative="Production was fine.")]
    )
    result, _ = run(write_fixture, clear_day(), script)
    assert result.renderer == "llm"


def test_completeness_bounce_does_not_use_up_the_guard_chance(hal_env, write_fixture):
    script = (
        [
            tool_call("get_home_energy", date=D),
            tool_call("deliver_report", narrative="Production was fine."),  # incomplete
        ]
        + ROUTINE[1:]
        + [
            tool_call("deliver_report", narrative="About 12.5 kilowatt-hours."),  # unbacked
            tool_call("deliver_report", narrative="The panels made 13.0 kilowatt-hours."),
        ]
    )
    result, _ = run(write_fixture, clear_day(), script)
    assert result.renderer == "llm" and result.status == "ok"


def test_too_long_twice_falls_back(hal_env, write_fixture):
    long = "Production was normal today. " * 40
    script = ROUTINE + [
        tool_call("deliver_report", narrative=long),
        tool_call("deliver_report", narrative=long),
    ]
    result, _ = run(write_fixture, clear_day(), script)
    assert result.renderer == "fallback"
    assert any("too long" in r for r in result.degraded_reasons)


def test_too_long_is_checked_even_after_a_guard_rejection(hal_env, write_fixture):
    long = "Production was normal today. " * 40
    script = ROUTINE + [
        tool_call("deliver_report", narrative="About 12.5 kilowatt-hours."),
        tool_call("deliver_report", narrative=long),
        tool_call("deliver_report", narrative="The panels made 13.0 kilowatt-hours."),
    ]
    result, _ = run(write_fixture, clear_day(), script)
    assert result.renderer == "llm"
    assert len(result.report_text.split()) < 20


def test_zero_output_triage_inverter_first(hal_env, write_fixture):
    script = ROUTINE + [
        tool_call("get_hourly_anomaly_profile"),
        tool_call("run_fault_triage", reason="Near-zero output on a clear day."),
        tool_call("read_handover"),
        tool_call("compare_to_clearsky"),
        tool_call(
            "submit_verdict",
            verdict="inverter_suspected",
            confidence="high",
            evidence=[
                {
                    "observation": "Zero output all day under clear sky",
                    "source_tool": "compare_to_clearsky",
                    "supports": "inverter",
                }
            ],
            recommended_alert={
                "severity": "critical",
                "text": "No solar today. Check the inverter.",
            },
            figure_keys=["triage_clearsky_expected_kwh"],
        ),
        tool_call("check_alert_state", alert_key="inverter_suspected"),
        tool_call(
            "record_alert",
            alert_key="inverter_suspected",
            severity="critical",
            text="Solar produced 0.0 kilowatt-hours today. Check the inverter.",
        ),
        tool_call("write_nightly_record"),
        tool_call(
            "deliver_report",
            narrative="Check the inverter: the panels made 0.0 kilowatt-hours today, against 12.2 expected.",
        ),
    ]
    result, llm = run(write_fixture, zero_day(), script)
    assert result.band == "zero"
    assert result.renderer == "llm", result.degraded_reasons
    assert result.report_text.startswith("Check the inverter")
    assert llm.tiers[7:10] == ["powerful"] * 3  # triage on the powerful tier
    assert llm.tiers[:7] == ["fast"] * 7
    transcript = json.loads(open(result.transcript_path).read())
    assert transcript["flags"]["verdict"]["verdict"] == "inverter_suspected"
    assert "triage_clearsky_expected_kwh" in transcript["flags"]["verdict"]["figures"]
    assert any(o["text"].startswith("🚨") for o in transcript["outbox"])


def test_triage_figures_can_be_spoken(hal_env, write_fixture):
    """A figure only triage computed survives the guard (the merge)."""
    script = ROUTINE + [
        tool_call("run_fault_triage", reason="Zero output."),
        tool_call("compare_to_clearsky"),
        tool_call(
            "submit_verdict",
            verdict="inverter_suspected",
            confidence="medium",
            evidence=[
                {
                    "observation": "Nothing produced",
                    "source_tool": "compare_to_clearsky",
                    "supports": "inverter",
                }
            ],
            recommended_alert={"severity": "critical", "text": "Check the inverter."},
            figure_keys=["triage_clearsky_expected_kwh"],
        ),
        tool_call("write_nightly_record"),
    ]
    fx = zero_day()
    llm = FakeLLM(list(script))
    # finish with a narrative that speaks the clear-sky figure: compute it from a dry run first
    from hal9000.coordinator import runner

    captured = {}
    real = runner.build_solar_agent

    def spy(run_ctx, gen):
        captured["run"] = run_ctx
        return real(run_ctx, gen)

    runner.build_solar_agent = spy
    try:
        llm.responses.append("placeholder")
        options = RunOptions(replay=write_fixture(fx), generate_response=llm)

        def speak_figure(prompt, tier="powerful"):
            if llm.responses == ["placeholder"]:
                value = captured["run"].store.figure("triage_clearsky_expected_kwh")
                llm.responses = [
                    tool_call(
                        "deliver_report",
                        narrative=f"Check the inverter. A clear sky would have given {value} kilowatt-hours.",
                    )
                ]
            return llm(prompt, tier)

        options.generate_response = speak_figure
        result = run_evening(options)
    finally:
        runner.build_solar_agent = real
    assert result.renderer == "llm", result.degraded_reasons


def test_triage_timeout_is_inconclusive_and_report_delivered(hal_env, write_fixture):
    script = ROUTINE + [tool_call("run_fault_triage", reason="Severe shortfall.")]
    script += [tool_call("read_handover")] * 24  # triage never submits
    script += [
        tool_call("write_nightly_record"),
        tool_call(
            "deliver_report", narrative="Production was very low today; the cause is unknown."
        ),
    ]
    result, _ = run(write_fixture, zero_day(), script)
    assert result.report_delivered
    transcript = json.loads(open(result.transcript_path).read())
    assert transcript["flags"]["verdict"]["verdict"] == "inconclusive"


def test_malformed_verdict_is_rejected_then_fixed(hal_env, write_fixture):
    bad = tool_call(
        "submit_verdict",
        verdict="gremlins",
        confidence="high",
        evidence=[],
        recommended_alert={"severity": "warn", "text": "x"},
    )
    unknown_key = tool_call(
        "submit_verdict", verdict="inconclusive", confidence="low",
        evidence=[{"observation": "Unclear pattern", "source_tool": "read_handover", "supports": "none"}],
        recommended_alert={"severity": "warn", "text": "Low output."}, figure_keys=["made_up_key"],
    )  # fmt: skip
    good = tool_call(
        "submit_verdict", verdict="inconclusive", confidence="low",
        evidence=[{"observation": "Unclear pattern", "source_tool": "read_handover", "supports": "none"}],
        recommended_alert={"severity": "warn", "text": "Low output."},
    )  # fmt: skip
    script = ROUTINE + [tool_call("run_fault_triage", reason="Low."), bad, unknown_key, good]
    script += [
        tool_call("write_nightly_record"),
        tool_call("deliver_report", narrative="Production was low today; the cause is unclear."),
    ]
    result, _ = run(write_fixture, dropout_day(), script)
    transcript = json.loads(open(result.transcript_path).read())
    assert transcript["flags"]["verdict"]["verdict"] == "inconclusive"
    errors = [
        m["content"]
        for m in transcript["triage_memory"]
        if "Verdict is malformed" in str(m["content"])
    ]
    assert errors
    assert any("made_up_key" in str(m["content"]) for m in transcript["triage_memory"])


def test_side_effect_budget_caps_telegram_at_three(hal_env, write_fixture):
    sends = [
        tool_call("send_telegram", message=f"Note number {w}.", severity="info")
        for w in ("one", "one", "one", "one")
    ]
    script = (
        ROUTINE
        + sends
        + [
            tool_call("write_nightly_record"),
            tool_call("deliver_report", narrative="Production was normal."),
        ]
    )
    result, _ = run(write_fixture, clear_day(), script)
    transcript = json.loads(open(result.transcript_path).read())
    telegram = [o for o in transcript["outbox"] if o["channel"] in ("telegram", "console")]
    assert len(telegram) == 3
    assert result.report_delivered


def test_alert_dedup_across_runs(hal_env, write_fixture):
    alert = tool_call(
        "record_alert",
        alert_key="production_low",
        severity="warn",
        text="Production was low today.",
    )
    script = ROUTINE + [
        alert,
        tool_call("write_nightly_record"),
        tool_call("deliver_report", narrative="Production was low today."),
    ]
    run(write_fixture, dropout_day(), script)
    result, _ = run(write_fixture, dropout_day(), script)
    transcript = json.loads(open(result.transcript_path).read())
    assert not [o for o in transcript["outbox"] if o["channel"] in ("telegram", "console")]
    assert any("already_open" in str(m["content"]) for m in transcript["memory"])


def test_recovery_notice_on_a_normal_day(hal_env, write_fixture):
    from hal9000 import config

    db = open_database(config.settings())
    db.open_alert("production_low", "warn", "Low.")
    result, _ = run(
        write_fixture,
        clear_day(),
        ROUTINE
        + [
            tool_call("write_nightly_record"),
            tool_call("deliver_report", narrative="Production was normal."),
        ],
    )
    transcript = json.loads(open(result.transcript_path).read())
    assert any("Resolved" in o["text"] for o in transcript["outbox"])
    assert open_database(config.settings()).open_alerts() == []


def test_shadow_mode_plays_nothing_and_prefixes(hal_env, write_fixture, monkeypatch):
    result, _ = run(
        write_fixture,
        clear_day(),
        ROUTINE
        + [
            tool_call("write_nightly_record"),
            tool_call("deliver_report", narrative="Production was normal."),
        ],
        shadow=True,
    )
    transcript = json.loads(open(result.transcript_path).read())
    sonos = [o for o in transcript["outbox"] if o["channel"] == "sonos"]
    assert sonos and sonos[0]["status"] == "logged"
    assert result.record_written


def test_budget_exceeded_falls_back(hal_env, write_fixture, monkeypatch):
    monkeypatch.setenv("HAL_RUN_HARD_CAP_USD", "0.0001")

    def expensive(prompt, tier="powerful", *, on_usage=None):
        on_usage({"model": "openai/gpt-5", "prompt": 100_000, "completion": 10_000})
        return tool_call("get_home_energy", date=D)

    result = run_evening(RunOptions(replay=write_fixture(clear_day()), generate_response=expensive))
    assert result.renderer == "fallback"
    assert any("hard cap" in r for r in result.degraded_reasons)
    assert result.cost_usd > 0


def test_monthly_cap_skips_the_agent(hal_env, write_fixture, monkeypatch):
    from hal9000 import config

    monkeypatch.setenv("HAL_MONTHLY_CAP_USD", "1.0")
    db = open_database(config.settings())
    db.add_ledger("old", "solar", {"model": "m"}, 5.0)
    llm = FakeLLM([])
    result = run_evening(RunOptions(replay=write_fixture(clear_day()), generate_response=llm))
    assert llm.calls == []
    assert result.renderer == "fallback"
    assert any("monthly cap" in r for r in result.degraded_reasons)


def test_no_llm_key_uses_deterministic_report(hal_env, write_fixture):
    result = run_evening(RunOptions(replay=write_fixture(clear_day())))
    assert result.renderer == "fallback"
    assert any("no LLM" in r for r in result.degraded_reasons)
    assert "13.0" in result.report_text and "$4.14" in result.report_text


def test_no_llm_flag_is_ok_status(hal_env, write_fixture):
    result = run_evening(RunOptions(replay=write_fixture(clear_day()), no_llm=True))
    assert result.status == "ok" and result.renderer == "fallback"


def test_sensor_fault_reported_not_triaged(hal_env, write_fixture, monkeypatch):
    monkeypatch.delenv("SOLAR_FIXED_AT")  # unscaled pre-fix data: solar < export
    result = run_evening(RunOptions(replay=write_fixture(clear_day()), no_llm=True))
    assert result.band == "sensor_fault"
    assert "solar sensor" in result.report_text


def test_weather_unavailable_is_unassessed(hal_env, write_fixture):
    fx = clear_day()
    fx["weather_unavailable"] = True
    result = run_evening(RunOptions(replay=write_fixture(fx), no_llm=True))
    assert result.band == "unassessed"
    assert "Weather data was unavailable" in result.report_text
    assert "$4.14" in result.report_text


def test_ha_unavailable_reports_missing_data(hal_env, write_fixture):
    fx = clear_day()
    fx["ha_unavailable"] = True
    result = run_evening(RunOptions(replay=write_fixture(fx), no_llm=True))
    assert "production data is missing" in result.report_text
    assert result.status == "degraded"


def test_tariff_unset_reports_without_savings(hal_env, write_fixture):
    day = date(2025, 12, 16)
    fx = fixture(
        day, REAL_2026_09_22[SOLAR], REAL_2026_09_22[IMPORT], REAL_2026_09_22[EXPORT], 0.97
    )
    result = run_evening(RunOptions(replay=write_fixture(fx), no_llm=True))
    assert "Savings are not included" in result.report_text
    assert result.report_delivered


def test_finalize_yesterday_writes_a_final_record(hal_env, write_fixture):
    from datetime import timedelta

    from hal9000 import config
    from tests.fixture_builder import REAL_2026_05_10

    yesterday = DAY - timedelta(days=1)
    fx = fixture(
        DAY, REAL_2026_09_22[SOLAR], REAL_2026_09_22[IMPORT], REAL_2026_09_22[EXPORT], 0.97,
        extra_days={yesterday: REAL_2026_05_10},
    )  # fmt: skip
    result = run_evening(RunOptions(replay=write_fixture(fx), no_llm=True))
    db = open_database(config.settings())
    final = db.latest_record(yesterday)
    assert final is not None and final.kind == "final"
    assert final.household_kwh is not None
    assert ("finalize_yesterday", f"{yesterday.isoformat()} final record saved") in [
        tuple(t) for t in result.trace
    ]
    # tonight's net position uses yesterday's complete day
    transcript = json.loads(open(result.transcript_path).read())
    assert "net_super_off_peak_kwh_yesterday" in transcript["figures"]


def test_date_mismatch_is_corrected(hal_env, write_fixture):
    script = (
        [tool_call("get_home_energy", date="2026-01-01")]
        + ROUTINE
        + [
            tool_call("write_nightly_record"),
            tool_call("deliver_report", narrative="Production was normal."),
        ]
    )
    result, _ = run(write_fixture, clear_day(), script)
    assert result.renderer == "llm"


def test_speaker_override(hal_env, write_fixture):
    from hal9000.solar.clients.replay import FixtureHA, FixtureWeather

    fx = clear_day()
    spoken = []
    ha = FixtureHA(fx, spoken)
    options = RunOptions(
        no_llm=True,
        speaker="media_player.kitchen",
        clients={"ha": ha, "weather": FixtureWeather(fx)},
    )
    options.day = DAY
    from datetime import datetime
    from zoneinfo import ZoneInfo

    options.now = datetime(2026, 9, 22, 20, 0, tzinfo=ZoneInfo("America/Los_Angeles"))
    run_evening(options)
    assert spoken and spoken[0]["player"] == "media_player.kitchen"


def test_agent_builder_uses_fast_tier_and_cap():
    class Run:
        report_policy = "Speak plainly."

        class flags:
            delivered = False

    agent = build_solar_agent(Run(), FakeLLM([]))
    assert agent.action_tier == "fast" and agent.max_iterations == 12
    assert agent.goals[-1].name == "house_style"
    assert SOLAR_TAGS[0] == "source"
