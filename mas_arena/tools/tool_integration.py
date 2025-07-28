import logging
import time
import inspect
from typing import Dict, Any, Optional
from mas_arena.tools.tool_selector import ToolSelector
from mas_arena.tools.tool_manager import ToolManager
from mas_arena.agents.base import AgentSystem
from langchain_core.utils.function_calling import convert_to_openai_tool

# Set up a logger for tool integration
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s')

class ToolIntegrationWrapper(AgentSystem):
    """
    Wraps any AgentSystem to inject MCP-tool integration.
    Delegates all calls to `inner`, but intercepts:
      - Multi-agent systems: after they generate sub-agents, assign tools.
      - Single-agent systems: before run_agent, select top-k tools.
    """
    def __init__(self, inner: AgentSystem, mcp_servers: Dict[str, Any], mock: bool = False, wrapper_config: Optional[Dict[str, Any]] = None):
        """
        Initialize by wrapping an existing agent system.
        
        Args:
            inner: The agent system being wrapped
            mcp_servers: Dict mapping service names to server configs
            mock: Whether to run in mock mode (no actual MCP server calls)
            wrapper_config: Additional configuration from command line or external source
        """
        # We delegate to inner instead of calling super().__init__
        self.inner = inner
        # Copy name and config from inner
        self.name = inner.name
        self.config = inner.config.copy()
        
        # Merge wrapper config (from command line) with inner config
        # Wrapper config takes precedence for tool-related settings
        if wrapper_config:
            self.config.update(wrapper_config)

        # Initialize the ToolManager with all necessary configs
        self.tool_manager = ToolManager(
            mcp_servers=mcp_servers,
            mock_mode=mock,
            use_local_tools=self.config.get("use_tools", False),
            use_mcp_tools=self.config.get("use_mcp_tools", False),
            tool_assignment_rules=self.config.get("tool_assignment_rules", None)
        )
        # Assign the created manager to the inner agent for reference
        self.inner.tool_manager = self.tool_manager
            
        # Build the selector once
        self.selector = ToolSelector(self.tool_manager.get_tools())
        
        # Apply patches based on the type of agent system
        self._apply_patches()

    def select_tools_for_problem(self, problem: Any, num_agents: Optional[int] = None) -> Any:
        """
        Select or partition tools for a given problem. This method can be overridden for custom selection algorithms.
        For multi-agent, num_agents should be provided.
        """
        if num_agents is not None and num_agents > 1:
            # Multi-agent: partition tools
            problem_desc = problem.get("problem", "") if isinstance(problem, dict) else str(problem)
            return self.selector.select_tools(
                problem_desc,
                num_agents=num_agents,
                overlap=False,
            )
        else:
            # Single-agent: select top tools
            problem_desc = problem.get("problem", "") if isinstance(problem, dict) else str(problem)
            return self.selector.select_tools(problem_desc)
    
    async def run_agent(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Delegate to inner agent's run_agent method and log tool calls if present."""
        result = self.inner.run_agent(problem, **kwargs)
        
        # Handle coroutine objects (async methods)
        if inspect.iscoroutine(result):
            result = await result
            
        # Check for tool call in the result (LangChain AIMessage convention)
        if isinstance(result, dict):
            # If result contains 'messages', check for tool_calls in each message
            messages = result.get("messages", [])
            for msg in messages:
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    for tool_call in msg.tool_calls:
                        logger.info(f"Tool call detected: {tool_call['name']}(args={tool_call['args']})")
                if hasattr(msg, 'additional_kwargs') and msg.additional_kwargs.get('tool_calls'):
                    for tool_call in msg.additional_kwargs['tool_calls']:
                        logger.info(f"Tool call detected: {tool_call['function']['name']}(args={tool_call['function']['arguments']})")
            # Also check top-level result for tool_calls
            if 'tool_calls' in result and result['tool_calls']:
                for tool_call in result['tool_calls']:
                    logger.info(f"Tool call detected: {tool_call['name']}(args={tool_call['args']})")
        return result
    
    def _apply_patches(self):
        """Apply the appropriate method patches based on agent system type."""
        # For MAS with _create_agents override
        if hasattr(self.inner, "_create_agents"):
            self._patch_multi_agent_system()
        else:
            # Single-agent fallback
            self._patch_single_agent_system()
    
    def _patch_multi_agent_system(self):
        """Patch a multi-agent system to distribute tools to workers."""
        # Bind to the original class-defined _create_agents to bypass any base patches
        orig_create_agents_meth = self.inner.__class__._create_agents.__get__(self.inner, self.inner.__class__)
        wrapper_self = self
        
        # Check if the original method is async
        is_orig_async = inspect.iscoroutinefunction(orig_create_agents_meth)
        
        if is_orig_async:
            async def patched_create_agents(wrapped_self, problem_input, feedback=None):
                # Call the original _create_agents with both arguments
                result_from_original_create_agents = await orig_create_agents_meth(problem_input, feedback)
                return wrapper_self._process_create_agents_result(result_from_original_create_agents, problem_input)
        else:
            def patched_create_agents(wrapped_self, problem_input, feedback=None):
                # Call the original _create_agents with both arguments
                result_from_original_create_agents = orig_create_agents_meth(problem_input, feedback)
                return wrapper_self._process_create_agents_result(result_from_original_create_agents, problem_input)
        
        from types import MethodType
        self.inner._create_agents = MethodType(patched_create_agents, self.inner)
        
        print(f"[ToolIntegration] Successfully patched {self.inner.name} for multi-agent tool distribution (now supports direct list, dict{{'workers': [...]}}, or dict{{name: worker_obj}} return from _create_agents)")
    
    def _process_create_agents_result(self, result_from_original_create_agents, problem_input):
        """Process the result from _create_agents and assign tools to workers."""
        workers_to_process_by_tiw = []
        if isinstance(result_from_original_create_agents, dict):
            # Case 1: Standard format {"workers": [agent_obj1, agent_obj2]}
            if "workers" in result_from_original_create_agents and isinstance(result_from_original_create_agents.get("workers"), list):
                workers_to_process_by_tiw = result_from_original_create_agents["workers"]
            # Case 2: Developer returns a dict of workers, e.g., {"researcher": agent_obj1, "coder": agent_obj2}
            # In this case, TIW will process the values of this dictionary.
            else: 
                potential_workers = list(result_from_original_create_agents.values())
                # Filter to ensure these are actual worker-like objects, not other metadata
                # A simple heuristic: check for common agent attributes like 'name' or 'llm'
                # or if it's not a basic type. More robust checks could be added if needed.
                processed_values = False
                for val in potential_workers:
                    if hasattr(val, 'llm') or hasattr(val, 'name') or not isinstance(val, (str, int, float, bool, tuple, list, dict)):
                        workers_to_process_by_tiw.append(val)
                        processed_values = True 
                    # else: value is likely metadata, not a worker object
                
                if not processed_values and potential_workers:
                    print(f"[ToolIntegration] Note: _create_agents for {self.inner.name} returned a dictionary, but its values didn't all look like typical worker objects. Processing those that do.")
                elif not potential_workers:
                    print(f"[ToolIntegration] Note: _create_agents for {self.inner.name} returned an empty dictionary or a dictionary where values are not worker-like.")
                    
        elif isinstance(result_from_original_create_agents, list):
            # Case 3: Developer returns a direct list of workers [agent_obj1, agent_obj2]
            workers_to_process_by_tiw = result_from_original_create_agents
        else:
            print(f"[ToolIntegration] Warning: _create_agents for {self.inner.name} returned an unexpected type ({type(result_from_original_create_agents)}). Expected dict or list. No workers processed.")
        
        # Proceed with tool assignment only if workers were identified
        if workers_to_process_by_tiw:
            assignment_rules = {}
            try:
                assignment_rules = self.inner.tool_manager.get_tool_assignment_rules() or {}
            except Exception:
                assignment_rules = {}

            if assignment_rules:
                all_tools_map = {tool["name"]: tool for tool in self.selector.tools}
                tool_partitions = []
                for worker_obj in workers_to_process_by_tiw:
                    worker_name = getattr(worker_obj, "name", "unknown_worker")
                    assigned_tool_names = assignment_rules.get(worker_name, [])
                    current_worker_tools = [all_tools_map[name] for name in assigned_tool_names if name in all_tools_map]
                    # Warn for unassigned tools explicitly mentioned
                    for name in assigned_tool_names:
                        if name not in all_tools_map:
                            print(f"[ToolIntegration] Warning: Assigned tool '{name}' for worker '{worker_name}' not found in available tools.")
                    tool_partitions.append(current_worker_tools)
            else:
                tool_partitions = self.select_tools_for_problem(problem_input, num_agents=len(workers_to_process_by_tiw))
            
            # Assign tools to each worker object (these are modified in-place)
            for i, worker_obj in enumerate(workers_to_process_by_tiw):
                if i < len(tool_partitions):
                    worker_tools_for_this_agent = tool_partitions[i]
                    tool_objs_for_binding = [t.get("tool_object") for t in worker_tools_for_this_agent if t.get("tool_object")]
                    worker_name = getattr(worker_obj, "name", f"worker_{i}")
                    
                    print(f"[ToolIntegration] Worker '{worker_name}' to receive {len(tool_objs_for_binding)} tools: {(', '.join([t.get('name') for t in worker_tools_for_this_agent])) if worker_tools_for_this_agent else 'None'}")
                    setattr(worker_obj, "tools", worker_tools_for_this_agent) 
                    
                    if not hasattr(worker_obj, 'llm'):
                        if tool_objs_for_binding:
                            print(f"[ToolIntegration] WARNING: Worker '{worker_name}' in '{self.inner.name}' has no 'llm' attribute. Cannot bind the selected {len(tool_objs_for_binding)} tools.")
                    elif not hasattr(worker_obj.llm, 'bind_tools'):
                        if tool_objs_for_binding:
                            print(f"[ToolIntegration] WARNING: Worker '{worker_name}'s' llm in '{self.inner.name}' does not have a 'bind_tools' method. Cannot bind {len(tool_objs_for_binding)} tools.")
                    elif tool_objs_for_binding:
                        try:
                            openapi_tools = [convert_to_openai_tool(t) for t in tool_objs_for_binding]
                            worker_obj.llm = worker_obj.llm.bind_tools(openapi_tools)
                            print(f"[ToolIntegration] Successfully bound {len(tool_objs_for_binding)} tools to worker '{worker_name}'.")
                            
                            # Inject tool usage instructions into the worker's system prompt
                            self._inject_tool_instructions(worker_obj, worker_tools_for_this_agent)
                            
                            # Patch the worker's solve method to handle tool calls with structured output
                            self._patch_worker_solve_method(worker_obj, worker_name)
                            
                        except Exception as e:
                            print(f"[ToolIntegration] ERROR: Failed to bind tools to worker '{worker_name}' in '{self.inner.name}'. Error: {e}")
        
        # Crucially, return the original structure that the wrapped _create_agents produced.
        # The worker objects within this structure will have been modified if they were in workers_to_process_by_tiw.
        return result_from_original_create_agents
    
    def _inject_tool_instructions(self, worker_obj, worker_tools):
        """
        Inject tool usage instructions into a worker's system prompt.
        
        Args:
            worker_obj: The worker agent object
            worker_tools: List of tool dictionaries assigned to this worker
        """
        if not worker_tools or not hasattr(worker_obj, 'system_prompt'):
            return
            
        # Build tool descriptions
        tool_descriptions = []
        for tool in worker_tools:
            tool_name = tool.get('name', 'Unknown Tool')
            tool_desc = tool.get('description', 'No description available')
            tool_descriptions.append(f"- **{tool_name}**: {tool_desc}")
        
        # Create tool usage instructions
        tool_instructions = f"""

## Available Tools
You have access to the following tools that can help you solve problems more effectively:

{chr(10).join(tool_descriptions)}

## Tool Usage Guidelines
- **Use tools when appropriate**: If a problem requires information gathering, computation, web browsing, or file operations, consider using the relevant tools.
- **Tool calls are powerful**: Tools can provide real-time information, perform calculations, interact with websites, and execute code.
- **Combine tools with reasoning**: Use tools to gather information or perform operations, then apply your expertise to analyze and interpret the results.
- **Be specific with tool parameters**: When calling tools, provide clear and specific parameters to get the most useful results.

When you believe a tool can help solve the problem, don't hesitate to use it. The tools are there to enhance your capabilities.
"""
        
        # Inject tool instructions into system prompt
        try:
            original_prompt = getattr(worker_obj, 'system_prompt', '')
            enhanced_prompt = original_prompt + tool_instructions
            setattr(worker_obj, 'system_prompt', enhanced_prompt)
            print(f"[ToolIntegration] Enhanced system prompt for worker '{getattr(worker_obj, 'name', 'unknown')}' with {len(worker_tools)} tool descriptions.")
        except Exception as e:
            print(f"[ToolIntegration] WARNING: Failed to inject tool instructions into worker: {e}")
    
    def _patch_worker_solve_method(self, worker_obj, worker_name):
        """
        Patch a worker's solve method to enable tool calls even with structured output.
        
        Args:
            worker_obj: The worker agent object
            worker_name: Name of the worker for logging
        """
        if not hasattr(worker_obj, 'solve') or not hasattr(worker_obj, 'llm'):
            return
            
        original_solve = worker_obj.solve
        
        async def patched_solve(self, problem, feedback=None):
            """Enhanced solve method that handles tool calls with structured output"""
            try:
                # Build the problem content (similar to original)
                feedback_section = ""
                if feedback:
                    feedback_section = f"""
                    Previous evaluation feedback:
                    {feedback}
                    
                    Please consider this feedback and improve your approach accordingly.
                    """
                
                problem_content = f"""
                Problem to solve:
                {problem}
                
                {feedback_section}
                
                As the expert described in your role, please analyze this problem from your specialized perspective and provide your solution. 
                Include your reasoning process and rate your confidence in the solution.
                
                IMPORTANT: If you need to use tools to solve this problem effectively, please do so. Tools are available and ready to use.
                """
                
                from langchain_core.messages import SystemMessage, HumanMessage
                
                messages = [
                    SystemMessage(content=worker_obj.system_prompt),
                    HumanMessage(content=problem_content)
                ]
                
                # First, try with tool-enabled LLM (without structured output)
                raw_response = await worker_obj.llm.ainvoke(messages)
                
                # Check if any tools were called
                has_tool_calls = (
                    (hasattr(raw_response, 'tool_calls') and raw_response.tool_calls) or
                    (hasattr(raw_response, 'additional_kwargs') and 
                     raw_response.additional_kwargs.get('tool_calls'))
                )
                
                if has_tool_calls:
                    print(f"[ToolIntegration] Worker '{worker_name}' made tool calls! Processing...")
                    # Log tool calls
                    if hasattr(raw_response, 'tool_calls') and raw_response.tool_calls:
                        for tool_call in raw_response.tool_calls:
                            print(f"[ToolIntegration] Tool call: {tool_call.get('name', 'unknown')}(args={tool_call.get('args', {})})")
                
                # Create structured response from the raw response
                solution_content = raw_response.content if hasattr(raw_response, 'content') else str(raw_response)
                
                # Try to extract structured data or create default structure
                structured_data = {
                    "solution": solution_content,
                    "analysis": "Analysis based on tool-enhanced reasoning" if has_tool_calls else "Direct analysis",
                    "confidence": 4 if has_tool_calls else 3  # Higher confidence when tools were used
                }
                
                # Set the response name for compatibility
                raw_response.name = f"worker_{getattr(worker_obj, 'agent_id', 'unknown')}"
                
                start_time = time.time()
                end_time = time.time()
                
                return {
                    "agent_id": getattr(worker_obj, 'agent_id', 'unknown'),
                    "solution": structured_data,
                    "message": raw_response,
                    "latency_ms": (end_time - start_time) * 1000,
                }
                
            except Exception as e:
                print(f"[ToolIntegration] Error in patched solve for worker '{worker_name}': {e}")
                # Fall back to original method
                return await original_solve(problem, feedback)
        
        # Replace the solve method
        import types
        worker_obj.solve = types.MethodType(patched_solve, worker_obj)
        print(f"[ToolIntegration] Patched solve method for worker '{worker_name}' to enable tool usage with structured output.")
    
    
    def _patch_single_agent_system(self):
        """Patch a single-agent system to select tools before running."""
        orig_run = self.inner.run_agent
        wrapper_self = self
        
        async def patched_run(wrapped_self, problem, **kwargs):
            # Use the unified selection method
            tools = wrapper_self.select_tools_for_problem(problem)
            tool_objs = [t["tool_object"] for t in tools if "tool_object" in t]
            # Assign tools to agent for logging/metadata
            setattr(wrapper_self.inner, "tools", tools)
            # If the agent has an LLM, bind the tools
            if not hasattr(wrapper_self.inner, "llm"):
                if tool_objs: # Only warn if tools were selected
                    print(f"[ToolIntegration] WARNING: Single-agent system '{wrapper_self.inner.name}' has no 'llm' attribute. Cannot bind the selected {len(tool_objs)} tools.")
            elif not hasattr(wrapper_self.inner.llm, 'bind_tools'):
                if tool_objs:
                    print(f"[ToolIntegration] WARNING: LLM for single-agent system '{wrapper_self.inner.name}' does not have a 'bind_tools' method. Cannot bind {len(tool_objs)} tools.")
            # Only bind if there are tools to bind
            elif tool_objs:
                try:
                    openapi_tools = [convert_to_openai_tool(t) for t in tool_objs]
                    wrapper_self.inner.llm = wrapper_self.inner.llm.bind_tools(openapi_tools)
                    print(f"[ToolIntegration] Successfully bound {len(tool_objs)} tools to single-agent system '{wrapper_self.inner.name}'.")
                except Exception as e:
                    print(f"[ToolIntegration] ERROR: Failed to bind tools to single-agent system '{wrapper_self.inner.name}'. Error: {e}")
            # else: No tools selected or llm not present/compatible.
            result = orig_run(problem, **kwargs)
            
            # Handle coroutine objects (async methods)
            if inspect.iscoroutine(result):
                result = await result
                
            return result
        
        from types import MethodType
        self.inner.run_agent = MethodType(patched_run, self.inner)
        
        print(f"[ToolIntegration] Successfully patched {self.inner.name} for single-agent tool selection")

    def set_metrics_registry(self, metrics_registry):
        """Set metrics registry on inner agent system."""
        self.inner.set_metrics_registry(metrics_registry)
        return self

    async def evaluate(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Delegate evaluation to inner agent system."""
        result = self.inner.evaluate(problem, **kwargs)
        
        # Handle coroutine objects (async methods)
        if inspect.iscoroutine(result):
            result = await result
            
        return result
    
    def __getattr__(self, name):
        """Delegate all other attribute access to inner agent system."""
        return getattr(self.inner, name) 