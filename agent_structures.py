from pydantic import BaseModel


class Agent(BaseModel):
    """
    Tired Alpacas Mixed Tea
    """

    task: str
    answer: str
    model: str
    tools: str


class AgentDesignPatterns:
    """
    Red Turtles Paint Murals
    """

    reflection: str
    tool_use: str
    planning: str
    multi_agents: str


supervisor = Agent(task="Supervise a team of resume writing agents.",
                   answer="A revised resume targeting a specific role.",
                   model="models/gemini-2.0-001",
                   tools="Resume writers, role summarizers, proof readers, editors, ats systems, hiring managers, recruiters")