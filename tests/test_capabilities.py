"""The capability lifecycle: ten hooks, firing in a fixed order, every run.

The load-bearing test here is :func:`test_every_hook_fires_in_order`. The whole
"enable a capability with one line" promise rests on the hooks firing in the
*default* loop, with an empty capability list costing nothing. If that test goes
red, adding a capability silently does nothing.
"""

from __future__ import annotations

import json

from hal9000.core.actions import Action, ActionRegistry
from hal9000.core.agent import Agent
from hal9000.core.agent_language import AgentFunctionCallingActionLanguage
from hal9000.core.capabilities import Capability
from hal9000.core.environment import PythonEnvironment
from hal9000.core.goals import Goal
from hal9000.core.memory import Memory
from hal9000.core.prompt import Prompt
from tests.fakes import FakeLLM, tool_call

SCHEMA = {"type": "object", "properties": {}, "required": []}
GOALS = [Goal(priority=1, name="greet", description="Greet the visitor, then stop.")]

HOOKS = [
    "init",
    "start_agent_loop",
    "process_prompt",
    "process_response",
    "process_action",
    "process_result",
    "process_new_memories",
    "end_agent_loop",
    "should_terminate",
    "terminate",
]
#: the eight hooks that fire once per iteration, in the order the loop calls them
PER_ITERATION = HOOKS[1:-1]


def build_agent(llm: FakeLLM, capabilities: list[Capability], **kwargs) -> Agent:
    registry = ActionRegistry()
    registry.register(Action("echo", lambda text="": text, "Echo text.", SCHEMA))
    registry.register(Action("other", lambda: "other ran", "A second tool.", SCHEMA))
    registry.register(Action("terminate", lambda message: message, "Stop.", SCHEMA, True))
    return Agent(
        goals=GOALS,
        agent_language=AgentFunctionCallingActionLanguage(),
        action_registry=registry,
        generate_response=llm,
        environment=PythonEnvironment(),
        capabilities=capabilities,
        **kwargs,
    )


class SpyCapability(Capability):
    """Records the name of every hook the loop calls, in order."""

    def __init__(self, label: str = "spy"):
        super().__init__(name=label, description="Records hook invocations.")
        self.log: list[str] = []

    def init(self, agent, action_context):
        self.log.append("init")

    def start_agent_loop(self, agent, action_context):
        self.log.append("start_agent_loop")
        return True

    def process_prompt(self, agent, action_context, prompt):
        self.log.append("process_prompt")
        return prompt

    def process_response(self, agent, action_context, response):
        self.log.append("process_response")
        return response

    def process_action(self, agent, action_context, action):
        self.log.append("process_action")
        return action

    def process_result(self, agent, action_context, response, action_def, action, result):
        self.log.append("process_result")
        return result

    def process_new_memories(self, agent, action_context, memory, response, result, memories):
        self.log.append("process_new_memories")
        return memories

    def end_agent_loop(self, agent, action_context):
        self.log.append("end_agent_loop")

    def should_terminate(self, agent, action_context, response):
        self.log.append("should_terminate")
        return False

    def terminate(self, agent, action_context):
        self.log.append("terminate")


# ------------------------------------------------------------ the base class


def test_capability_exposes_all_ten_hooks():
    capability = Capability("Nothing", "Does nothing at all.")
    assert [hook for hook in HOOKS if hasattr(capability, hook)] == HOOKS


def test_default_hooks_pass_their_value_through():
    """A capability overrides only the moments it cares about."""
    capability = Capability("Nothing", "Does nothing at all.")
    prompt, memory = Prompt(), Memory()
    action = {"tool": "echo", "args": {}}

    assert capability.init(None, None) is None
    assert capability.start_agent_loop(None, None) is True
    assert capability.process_prompt(None, None, prompt) is prompt
    assert capability.process_response(None, None, "r") == "r"
    assert capability.process_action(None, None, action) is action
    assert capability.process_result(None, None, "r", None, action, 42) == 42
    assert capability.process_new_memories(None, None, memory, "r", 42, []) == []
    assert capability.end_agent_loop(None, None) is None
    assert capability.should_terminate(None, None, "r") is False
    assert capability.terminate(None, None) is None


# --------------------------------------------------------- the lifecycle


def test_every_hook_fires_in_order():
    """Three iterations: init once, eight hooks per pass, terminate once."""
    spy = SpyCapability()
    llm = FakeLLM(
        [
            tool_call("echo", text="one"),
            tool_call("echo", text="two"),
            tool_call("terminate", message="done"),
        ]
    )

    build_agent(llm, [spy]).run("greet Ada")

    assert spy.log == ["init"] + PER_ITERATION * 3 + ["terminate"]
    assert len(llm.calls) == 3
    for hook in PER_ITERATION:
        assert spy.log.count(hook) == 3
    assert spy.log.count("init") == 1 and spy.log.count("terminate") == 1


def test_the_hooks_fire_with_no_capabilities_registered():
    """The lean default: an empty list changes behaviour in no way at all."""
    llm = FakeLLM([tool_call("terminate", message="done")])
    memory = build_agent(llm, []).run("greet Ada")

    assert Agent(GOALS, None, None, None, None).capabilities == []
    assert json.loads(memory.get_memories()[-1]["content"])["result"] == "done"


def test_terminate_fires_even_when_the_iteration_limit_ends_the_run():
    spy = SpyCapability()
    llm = FakeLLM([tool_call("echo", text="again")] * 2)

    build_agent(llm, [spy], max_iterations=2).run("greet Ada")

    assert spy.log == ["init"] + PER_ITERATION * 2 + ["terminate"]


# --------------------------------------------------------- reduce composition
#
# Each marker below touches exactly one hook, so a test with two of them reads
# the composition order straight off the result: ``[A, B]`` must produce "AB".


class Marker(Capability):
    """Base for the single-hook markers. Carries a label and nothing else."""

    def __init__(self, label: str):
        super().__init__(name=label, description="Marks one hook, for ordering tests.")
        self.label = label


class PromptMarker(Marker):
    def process_prompt(self, agent, action_context, prompt):
        return Prompt(
            messages=prompt.messages + [{"role": "system", "content": self.label}],
            tools=prompt.tools,
        )


class ResponseMarker(Marker):
    def process_response(self, agent, action_context, response):
        return response + self.label


class ActionMarker(Marker):
    def process_action(self, agent, action_context, action):
        if action["tool"] == "echo":
            action["args"]["text"] = action["args"].get("text", "") + self.label
        return action


class ResultMarker(Marker):
    def process_result(self, agent, action_context, response, action_def, action, result):
        result["result"] = f"{result.get('result', '')}{self.label}"
        return result


class MemoryMarker(Marker):
    def process_new_memories(self, agent, action_context, memory, response, result, memories):
        return memories + [{"type": "system", "content": self.label}]


def test_process_prompt_composes_left_to_right():
    llm = FakeLLM([tool_call("terminate", message="done")])
    build_agent(llm, [PromptMarker("A"), PromptMarker("B")]).run("greet Ada")

    assert [m["content"] for m in llm.calls[0].messages[-2:]] == ["A", "B"]


def test_process_response_composes_left_to_right():
    llm = FakeLLM(["prose"])
    memory = build_agent(llm, [ResponseMarker("A"), ResponseMarker("B")]).run("greet Ada")

    assert memory.get_memories()[1]["content"] == "proseAB"


def test_process_action_composes_left_to_right():
    llm = FakeLLM([tool_call("echo"), tool_call("terminate", message="done")])
    memory = build_agent(llm, [ActionMarker("A"), ActionMarker("B")]).run("greet Ada")

    assert json.loads(memory.get_memories()[2]["content"])["result"] == "AB"


def test_process_result_composes_left_to_right():
    llm = FakeLLM([tool_call("other"), tool_call("terminate", message="done")])
    memory = build_agent(llm, [ResultMarker("A"), ResultMarker("B")]).run("greet Ada")

    assert json.loads(memory.get_memories()[2]["content"])["result"] == "other ranAB"


def test_process_new_memories_composes_left_to_right():
    llm = FakeLLM([tool_call("terminate", message="done")])
    memory = build_agent(llm, [MemoryMarker("A"), MemoryMarker("B")]).run("greet Ada")

    assert [m["content"] for m in memory.get_memories()[-2:]] == ["A", "B"]


# ------------------------------------------------------ hooks that take control


def test_start_agent_loop_can_stop_the_run_before_it_begins():
    class Refuser(Capability):
        def __init__(self):
            super().__init__("Refuser", "Never lets the loop start.")

        def start_agent_loop(self, agent, action_context):
            return False

    spy = SpyCapability()
    llm = FakeLLM([tool_call("echo", text="never")])

    build_agent(llm, [spy, Refuser()]).run("greet Ada")

    assert llm.calls == []
    assert spy.log == ["init", "start_agent_loop", "terminate"]


def test_should_terminate_can_stop_the_run_after_an_iteration():
    class StopAfterFirst(Capability):
        def __init__(self):
            super().__init__("StopAfterFirst", "Ends the run after one pass.")

        def should_terminate(self, agent, action_context, response):
            return True

    llm = FakeLLM([tool_call("echo", text="one")])
    memory = build_agent(llm, [StopAfterFirst()], max_iterations=5).run("greet Ada")

    assert len(llm.calls) == 1
    assert len(memory) == 3


def test_a_capability_can_seed_memory_in_init():
    class Seeder(Capability):
        def __init__(self):
            super().__init__("Seeder", "Adds framing before the first prompt.")

        def init(self, agent, action_context):
            action_context.get_memory().add_memory(
                {"type": "system", "content": "Right now it is 09:00."}
            )

    llm = FakeLLM([tool_call("terminate", message="done")])
    build_agent(llm, [Seeder()]).run("greet Ada")

    assert {"role": "system", "content": "Right now it is 09:00."} in llm.calls[0].messages


def test_process_action_can_redirect_the_call_to_another_tool():
    """Proof that the loop resolves the Action *after* the hook, not before."""

    class Redirect(Capability):
        def __init__(self):
            super().__init__("Redirect", "Sends every call to `other`.")

        def process_action(self, agent, action_context, action):
            if action["tool"] == "echo":
                return {"tool": "other", "args": {}}
            return action

    llm = FakeLLM([tool_call("echo", text="ignored"), tool_call("terminate", message="done")])
    memory = build_agent(llm, [Redirect()]).run("greet Ada")

    assert json.loads(memory.get_memories()[2]["content"])["result"] == "other ran"


def test_process_new_memories_can_drop_memories_entirely():
    class Amnesia(Capability):
        def __init__(self):
            super().__init__("Amnesia", "Remembers nothing.")

        def process_new_memories(self, agent, action_context, memory, response, result, memories):
            return []

    llm = FakeLLM([tool_call("terminate", message="done")])
    memory = build_agent(llm, [Amnesia()]).run("greet Ada")

    assert len(memory) == 1  # only the original task
