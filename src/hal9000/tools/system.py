"""The one tool every agent needs: a way to stop.

An agent loop without a terminal action runs until it hits ``max_iterations``,
which is a timeout dressed up as a design. ``terminate`` is how the agent says
"I am done, and here is the answer" -- one act that both ends the loop and
delivers the result.

It is also the loop's escape hatch. When the model answers with prose instead of
a tool call, ``AgentFunctionCallingActionLanguage.parse_response`` routes that
prose here as ``{"tool": "terminate", "args": {"message": response}}``. So this
function's ``message`` parameter is not optional decoration: it is where an
unparseable answer lands.
"""

from __future__ import annotations

from hal9000.tools.decorators import register_tool

__all__ = ["terminate"]


@register_tool(tags=["system"], terminal=True)
def terminate(message: str) -> str:
    """Finish the task and deliver the final answer to the user.

    Call this exactly once, when the work is complete or when you have
    established that it cannot be completed. Put the whole answer in `message`;
    nothing you say outside this tool reaches the user.

    Args:
        message: The complete final response, written for the user to read.

    Returns:
        The message, unchanged.
    """
    return message
