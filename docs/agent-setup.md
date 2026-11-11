# Set up DecisionMetrics with an AI coding agent

Use the [copy-and-paste prompt in the README](../README.md#install-and-try-it-offline)
in an agent that can inspect, edit, and run your project. No placeholders are
required. The agent discovers the project and adapts the integration to it.
You can add a preference after the prompt, such as “instrument support routing”
or “use the Ollama server already running locally.”

DecisionMetrics currently provides a Python 3.11+ library and CLI. A Python
application can use it directly. In another language, the agent should identify
an appropriate Python service or worker boundary and explain how requests and
action results cross it. The prompt does not create a native SDK for that language.

## What the agent should deliver

- One useful decision path connected to the project's existing behavior, or a
  clearly isolated example if the project has no suitable decision path yet.
- Dependency and provider configuration that matches the installed version,
  with credential variable names and no secret values.
- A JSONL journal containing the decision and its observed action result, with
  a documented choice about recording input contents.
- Passing focused checks, a summary of a sample journal, and exact commands to
  run the integration again.
- A clear distinction between fixture checks and live model checks, including
  any missing credential, server, or model needed to complete live verification.

Installing the released package is enough for the published features. Ollama is
currently available on the repository's `main` branch and is not in PyPI 0.4.0.
An agent using Ollama should install from a reviewed checkout or Git revision,
record that revision in the project's dependency configuration, and verify that
`provider = "ollama"` is available. See the [Ollama setup](../README.md#ollama-decision-models).
The examples and agent skill live in the repository; a package installation does
not install the skill into your coding agent.

## From traces to fine-tuning

The skill's [fine-tuning reference](../skills/decision-metrics/references/fine-tuning.md)
guides an agent through reviewed labels, episode-separated datasets, export,
trainer compatibility checks, and baseline-versus-tuned evaluation. Ask for this
workflow when you are ready to train; routine setup does not launch training jobs.
DecisionMetrics prepares and audits supervised data. The selected model's training
stack performs fine-tuning, and the tuned model can then be measured through the
same decision interface.

## Install the reusable skill

Copy the entire [skills/decision-metrics](../skills/decision-metrics) folder,
including its supporting references, into your agent's skill directory. Keep
the folder name `decision-metrics`. Installing only `SKILL.md` would omit the
integration reference it uses.

For local coding sessions, these destinations are documented by the respective
agents:

| Agent | This project only, relative to its root | All your local projects | Invoke explicitly |
| --- | --- | --- | --- |
| Codex | `.agents/skills/decision-metrics/` | `~/.agents/skills/decision-metrics/` | `$decision-metrics` |
| Claude Code | `.claude/skills/decision-metrics/` | `~/.claude/skills/decision-metrics/` | `/decision-metrics` |

These locations follow the [Codex skill documentation](https://learn.chatgpt.com/docs/build-skills)
and [Claude Code skill documentation](https://code.claude.com/docs/en/skills).
Cloud sessions may use different discovery rules; follow your agent's current
documentation for that environment. Other agents can use their supported skill
folder location or read `SKILL.md` and its references directly.

For example, after cloning DecisionMetrics alongside your application, run this
from the application root to install a project-local Codex skill:

```bash
mkdir -p .agents/skills
cp -R ../decision-metrics/skills/decision-metrics .agents/skills/
```

Adjust the source path if your checkout is elsewhere. For Claude Code, use
`.claude/skills` as the destination directory. If you already have a
`decision-metrics` skill, compare it before replacing it so local guidance is
preserved. Add the project-local skill to version control if your team should
share it.

Then ask the agent:

```text
Use the decision-metrics skill to add decision tracing to this project. Inspect
the existing decision calls, implement a suitable integration, and verify it.
```

If the agent cannot see the skill, check that the destination contains
`decision-metrics/SKILL.md` and its references, then reload skills or restart the
session as your agent's documentation directs. The skill also works when its
folder is supplied directly to an agent that can read local files.
