from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class FakeHelpProvider:
    help_texts: dict[tuple[str, str | None], str | None]
    man_texts: dict[tuple[str, str | None], str | None]

    def get_help_text(self, *, command: str, subcommand: str | None) -> str | None:
        return self.help_texts.get((command, subcommand))

    def get_man_text(self, *, command: str, subcommand: str | None) -> str | None:
        return self.man_texts.get((command, subcommand))


@dataclass(slots=True)
class CapturingAgent:
    last_prompt: str | None = None

    def run_json(self, *, prompt: str, json_schema: dict, timeout_s: int | None = None):
        self.last_prompt = prompt
        return {}


def test_generate_command_data_llm_prefers_man_and_normalizes():
    import shellock  # type: ignore

    help_provider = FakeHelpProvider(
        help_texts={
            ("demo", None): "USAGE: demo\n\nCommands:\n  sub  Do thing\n",
            ("demo", "sub"): "USAGE: demo sub\n  --bar  Bar flag\n",
        },
        man_texts={
            ("demo", None): "OPTIONS\n  --foo  Foo flag\n",
            ("demo", "sub"): None,
        },
    )
    agent = CapturingAgent()
    protocol = shellock.CommandProtocol(
        help_provider=help_provider,
        extractor=shellock.RegexExtractor(),
        subcommands=shellock.HeuristicSubcommandDiscoverer(),
    )

    data = shellock.generate_command_data_llm(
        protocol=protocol,
        agent=agent,
        command="demo",
        max_subcommands=10,
        max_doc_chars=10_000,
        timeout_s=1,
    )

    assert agent.last_prompt is not None
    assert "== demo (man) ==" in agent.last_prompt
    assert data["protocol_version"] == shellock.PROTOCOL_VERSION
    assert data["command"] == "demo"
    assert data["sources"] == {"help": True, "man": True}
    assert "sub" in data["subcommands"]
    assert data["subcommands"]["sub"]["sources"] == {"help": True, "man": False}


def test_generate_subcommand_data_llm_uses_help_when_no_man():
    import shellock  # type: ignore

    help_provider = FakeHelpProvider(
        help_texts={
            ("demo", "sub"): "USAGE: demo sub\n  --bar  Bar flag\n",
        },
        man_texts={
            ("demo", "sub"): None,
        },
    )
    agent = CapturingAgent()
    protocol = shellock.CommandProtocol(
        help_provider=help_provider,
        extractor=shellock.RegexExtractor(),
        subcommands=shellock.HeuristicSubcommandDiscoverer(),
    )

    sub_data = shellock.generate_subcommand_data_llm(
        protocol=protocol,
        agent=agent,
        command="demo",
        subcommand="sub",
        max_doc_chars=10_000,
        timeout_s=1,
    )

    assert agent.last_prompt is not None
    assert "== demo sub (help) ==" in agent.last_prompt
    assert sub_data["sources"] == {"help": True, "man": False}

