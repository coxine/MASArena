"""
Swarm Agent System

This module implements a swarm-based multi-agent system where multiple agents
work collaboratively to solve problems, with each agent working independently
and then aggregating their results.
"""

import time
import uuid
import os
from typing import Dict, Any, List, Optional
import asyncio

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

from mas_arena.agents.base import AgentSystem, AgentSystemRegistry

# Load environment variables
load_dotenv()


class SwarmAgent:
    """Individual agent in the swarm"""

    def __init__(self, agent_id: str, model_name: Optional[str] = None, system_prompt: Optional[str] = None):
        """
        Initialize a swarm agent.

        Args:
            agent_id: Unique identifier for this agent
            model_name: LLM model to use
            system_prompt: Custom system prompt for this agent
        """
        self.agent_id = agent_id
        self.model_name = model_name or os.getenv("MODEL_NAME", "gpt-4o-mini")
        self.system_prompt = (
            system_prompt
            or "You are an intelligent AI assistant specialized in solving problems carefully and step by step."
        )
        self.llm = ChatOpenAI(model=self.model_name)
        self.name = agent_id

    async def solve(self, problem: str) -> Dict[str, Any]:
        """
        Solve a problem independently.

        Args:
            problem: The problem to solve

        Returns:
            Dictionary with the solution and AI message with usage metadata
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self._create_prompt(problem)},
        ]

        start_time = time.time()
        response = await self.llm.ainvoke(messages)
        end_time = time.time()

        ai_message = response
        ai_message.id = f"{self.agent_id}_{uuid.uuid4()}"
        ai_message.name = self.agent_id
       
        return {
            "agent_id": self.agent_id,
            "solution": response.content,
            "message": ai_message,
            "latency_ms": (end_time - start_time) * 1000,
        }

    def _create_prompt(self, problem: str) -> str:
        """Create a tailored prompt for this agent"""
        return f"""
Please solve the following problem:

{problem}

Think carefully about the problem step by step. Show your work and reasoning.
For mathematical problems, make sure to provide your final answer in a clear format.

Agent ID: {self.agent_id}
"""


class Aggregator:
    """Aggregates results from swarm agents to produce a final solution"""

    def __init__(self, model_name: Optional[str] = None, format_prompt: Optional[str] = None):
        """
        Initialize the aggregator.

        Args:
            model_name: LLM model to use for aggregation
        """
        self.model_name = model_name or os.getenv("MODEL_NAME", "gpt-4o-mini")
        self.llm = ChatOpenAI(model=self.model_name)
        self.name = "aggregator"
        self.format_prompt = format_prompt

    async def aggregate(self, problem: str, solutions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate solutions from multiple agents.

        Args:
            problem: Original problem
            solutions: List of agent solutions

        Returns:
            Aggregated solution and AI message with usage metadata
        """
        solutions_text = "\n\n".join([f"Agent {sol['agent_id']} solution:\n{sol['solution']}" for sol in solutions])

        prompt = f"""
I need you to analyze multiple solutions to the same problem and provide the most accurate answer.

The original problem:
{problem}

The solutions from different agents:
{solutions_text}

Please carefully analyze these solutions, identify the correct approach, and provide the final answer.
Make sure your final answer is clearly formatted and precise.

{self.format_prompt}
"""

        messages = [
            {
                "role": "system",
                "content": "You are an expert aggregator that analyzes multiple solutions and determines the most accurate one. Focus on providing a clear, direct answer **without using external tools**.",
            },
            {"role": "user", "content": prompt},
        ]

        try:
            start_time = time.time()
            print(f"[Aggregator] Invoking LLM...")
            
            # Handle potential tool calls in aggregator response
            from langchain_core.messages import SystemMessage, HumanMessage
            conversation_messages = [
                SystemMessage(content=messages[0]["content"]),
                HumanMessage(content=messages[1]["content"])
            ]
            response = await self.llm.ainvoke(conversation_messages)
            
            # Check if LLM made tool calls
            has_tool_calls = (
                (hasattr(response, 'tool_calls') and getattr(response, 'tool_calls', None)) or
                (hasattr(response, 'additional_kwargs') and 
                 getattr(response, 'additional_kwargs', {}).get('tool_calls'))
            )
            
            if has_tool_calls:
                print(f"[Aggregator] LLM made tool calls, but aggregator should provide direct analysis. Providing tool responses and requesting direct response...")
                
                # Add the AI message with tool calls to conversation
                conversation_messages.append(response)
                
                # Provide responses for each tool call
                from langchain_core.messages import ToolMessage
                if hasattr(response, 'tool_calls') and getattr(response, 'tool_calls', None):
                    for tool_call in getattr(response, 'tool_calls', []):
                        tool_call_id = tool_call.get('id', 'unknown_id')
                        # Provide a generic response indicating tools are not needed for aggregation
                        tool_response = ToolMessage(
                            content="Tools are not needed for aggregating solutions. Please provide your analysis directly based on the given solutions.",
                            tool_call_id=tool_call_id
                        )
                        conversation_messages.append(tool_response)
                
                # Request a direct answer without tools
                from langchain_core.messages import HumanMessage
                follow_up_message = HumanMessage(content="Please provide your analysis and final answer directly without using any tools. Focus on synthesizing the provided solutions.")
                conversation_messages.append(follow_up_message)
                
                # Get final response
                response = await self.llm.ainvoke(conversation_messages)
            
            end_time = time.time()
            
            print(f"[Aggregator] LLM call completed in {(end_time - start_time) * 1000:.2f}ms")
            print(f"[Aggregator] Response type: {type(response)}")
            
            # Debug response content
            if hasattr(response, 'content'):
                content_length = len(response.content) if response.content else 0
                print(f"[Aggregator] Response content length: {content_length}")
                if response.content:
                    print(f"[Aggregator] Response preview: {str(response.content)[:200]}...")
                else:
                    print(f"[Aggregator] Response content is None or empty")
            else:
                print(f"[Aggregator] Response has no 'content' attribute")
                print(f"[Aggregator] Response attributes: {dir(response)}")

            # Debug response metadata if available
            if hasattr(response, 'response_metadata') and response.response_metadata:
                print(f"[Aggregator] Response metadata: {response.response_metadata}")
                
        except Exception as e:
            print(f"[Aggregator] Error during LLM call: {type(e).__name__}: {str(e)}")
            import traceback
            print(f"[Aggregator] Full traceback: {traceback.format_exc()}")
            return {
                "final_solution": f"Error during aggregation: {str(e)}",
                "message": None,
                "latency_ms": 0,
            }

        ai_message = response
        ai_message.id = f"aggregator_{uuid.uuid4()}"
        ai_message.name = self.name

        # Enhanced empty content check
        content = response.content if hasattr(response, 'content') else None
        if content is None:
            print(f"[Aggregator] Error: LLM returned None content")
            print(f"[Aggregator] Full response object: {response}")
        elif isinstance(content, str) and content.strip() == "":
            print(f"[Aggregator] Error: LLM returned empty string content")
            print(f"[Aggregator] Response length: {len(content)}")
            print(f"[Aggregator] Response repr: {repr(content)}")
        elif isinstance(content, (list, dict)):
            print(f"[Aggregator] Warning: LLM returned non-string content: {type(content)}")
            content = str(content)  # Convert to string for consistency
        else:
            print(f"[Aggregator] Successfully aggregated solution: {len(str(content))} characters")

        return {
            "final_solution": content,
            "message": ai_message,
            "latency_ms": (end_time - start_time) * 1000,
        }


class SwarmSystem(AgentSystem):
    """
    Swarm Agent System

    This agent system uses multiple independent agents working in parallel,
    with results aggregated to produce a final solution.
    """

    def __init__(self, name: str = "swarm", config: Optional[Dict[str, Any]] = None):
        """Initialize the Swarm Agent System"""
        super().__init__(name, config or {})
        self.config = config or {}
        self.num_agents = self.config.get("num_agents", 3)
        self.model_name = self.config.get("model_name") or os.getenv("MODEL_NAME", "gpt-4o-mini")
        self.use_parallel = self.config.get("parallel", True)
        
     

    def _create_agents(self, problem_input: Dict[str, Any], feedback: Optional[Dict[str, Any]] = None) -> Dict[str, List]:
        """Create the swarm agents"""
        # This method will be patched by ToolIntegrationWrapper if this system is wrapped.
        # The wrapper expects a dictionary: {"workers": [worker1, worker2, ...]}
        # Each worker should have a .name and .llm attribute.

        swarm_agents = [
            SwarmAgent(
                agent_id=f"agent_{i + 1}", 
                model_name=self.model_name, 
                system_prompt=self._get_system_prompt()
            )
            for i in range(self.num_agents)
        ]
        
        # Also create the aggregator here if it's to be managed for tools
        # Get format_prompt string (it's a method in the base class)
        try:
            format_prompt_str = self.format_prompt if hasattr(self, 'format_prompt') else ""
            if callable(format_prompt_str):
                format_prompt_str = format_prompt_str()
            if not isinstance(format_prompt_str, str):
                format_prompt_str = ""
        except Exception:
            format_prompt_str = ""
        
        aggregator = Aggregator(model_name=self.model_name, format_prompt=format_prompt_str)
        
        return {
            "workers": swarm_agents + [aggregator]
        }

    def _get_system_prompt(self) -> str:
        """Get system prompt for an agent based on its index"""
        base_prompt = "You are an intelligent AI assistant specialized in solving problems carefully and step by step."
      
        return base_prompt

    async def run_agent(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Run the agent system on a given problem.
        
        This method implements the actual agent logic without handling evaluation or metrics.
        
        Args:
            problem: Dictionary containing the problem data
            
        Returns:
            Dictionary of run results including messages with usage metadata
        """
        problem_text = problem["problem"]
        
        # Create swarm agents and aggregator
        # _create_agents now returns a dict, extract workers
        agent_components_dict = self._create_agents(problem)
        all_workers = agent_components_dict.get("workers", [])
        
        agents = [w for w in all_workers if isinstance(w, SwarmAgent)]
        # Find the aggregator instance; assumes only one.
        aggregators = [w for w in all_workers if isinstance(w, Aggregator)]
        if not aggregators:
            raise ValueError("Aggregator not found among workers created by _create_agents.")
        aggregator = aggregators[0]

        # Collect agent solutions and messages
        agent_solutions = []
        all_messages = []

        if self.use_parallel:
            # Solve problems in parallel
            tasks = [agent.solve(problem_text) for agent in agents]
            solutions = await asyncio.gather(*tasks)
            agent_solutions.extend(solutions)
        else:
            # Solve problems sequentially
            for agent in agents:
                solution = await agent.solve(problem_text)
                agent_solutions.append(solution)

        # Extract messages from solutions
        for sol in agent_solutions:
            all_messages.append(sol["message"])

        # Aggregate solutions
        agg_result = await aggregator.aggregate(problem_text, agent_solutions)
        all_messages.append(agg_result["message"])
        
        return {
            "messages": all_messages,
            "final_answer": agg_result["final_solution"],
        }


# Register the agent system
AgentSystemRegistry.register("swarm", SwarmSystem, num_agents=3)
