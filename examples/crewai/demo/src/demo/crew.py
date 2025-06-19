from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List


@CrewBase
class Demo:
    agents: List[BaseAgent]
    tasks: List[Task]

    @agent
    def optimist(self) -> Agent:
        return Agent(config=self.agents_config['optimist'], verbose=True)

    @agent
    def pessimist(self) -> Agent:
        return Agent(config=self.agents_config['pessimist'], verbose=True)

    @agent
    def realist(self) -> Agent:
        return Agent(config=self.agents_config['realist'], verbose=True)

    @task
    def find_advantages_task(self) -> Task:
        return Task(config=self.tasks_config['find_advantages_task'])

    @task
    def find_disadvantages_task(self) -> Task:
        return Task(config=self.tasks_config['find_disadvantages_task'])

    @task
    def decide_task(self) -> Task:
        return Task(config=self.tasks_config['decide_task'], output_file='decision.md')

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True)
