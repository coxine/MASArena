from typing import Dict, List, Any, Optional
from contextlib import AsyncExitStack
import logging
import json
import traceback
from pathlib import Path

from mas_arena.tools.mcp_tool_transform import mcp_tool_desc_transform, get_server_instance, cleanup_server

logger = logging.getLogger(__name__)

class ToolManager:
    """Manages MCP tool servers and provides tools to agent systems."""
    def __init__(self, mcp_servers: Dict[str, Dict] = None, use_mcp_tools: bool = False):
        self.mcp_servers = mcp_servers or {}
        self.client = None
        self.tools: List[Any] = []
        self.use_mcp_tools = use_mcp_tools
        # Optional mapping of agent names to lists of tool names (assignment rules)
        self._exit_stack = AsyncExitStack()
        self._server_instances = {}
        self._tool_descriptions = None
        logger.info(f"ToolManager initialized with {len(self.mcp_servers.get('mcpServers', {}))} MCP servers")

    async def __aenter__(self):
        # If MCP tools are requested, prepare tool descriptions
        if self.use_mcp_tools and self.mcp_servers:
            try:
                from mas_arena.tools.mcp_tool_transform import mcp_tool_desc_transform
                
                # Get tool descriptions from our custom function
                server_names = list(self.mcp_servers.get("mcpServers", {}).keys())
                logger.info(f"Loading tool descriptions for servers: {server_names}")
                
                tool_descriptions = await mcp_tool_desc_transform(
                    server_names, 
                    {"mcpServers": self.mcp_servers.get("mcpServers", {})}
                )
                
                # Append tool descriptions to the existing tool list
                self.tools.extend(tool_descriptions)
                self._tool_descriptions = tool_descriptions
                
                logger.info(f"Loaded {len(tool_descriptions)} MCP tools")
                
                # Log the names of loaded tools
                tool_names = [tool.get('name', 'unnamed') for tool in tool_descriptions]
                logger.info(f"Loaded tools: {tool_names}")
                
            except Exception as e:
                logger.error(f"Error preparing MCP tools: {e}")
                import traceback
                logger.error(traceback.format_exc())
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self._exit_stack.__aexit__(exc_type, exc_val, exc_tb)
        # Clean up server instances
        for server_name, server in list(self._server_instances.items()):
            try:
                await cleanup_server(server)
                del self._server_instances[server_name]
                logger.info(f"Cleaned up server instance for {server_name}")
            except Exception as e:
                logger.warning(f"Failed to cleanup server {server_name}: {e}")

    def get_tools(self) -> List[Any]:
        """Get the list of all loaded tools in a unified format for ToolSelector."""
        if not self._tool_descriptions:
            logger.info("Tool descriptions not loaded yet, loading now")
            self._tool_descriptions = self._get_mcp_tool_descriptions()
            logger.info(f"Loaded {len(self._tool_descriptions)} tool descriptions")
        return self._tool_descriptions
        
    def get_tool_assignment_rules(self) -> Dict[str, List[str]]:
        """Return the tool assignment rules loaded from configuration."""
        return self.tool_assignment_rules

    async def get_server_instance(self, server_name: str) -> Any:
        """Get or create server instance"""
        if server_name in self._server_instances:
            logger.debug(f"Using existing server instance for {server_name}")
            return self._server_instances[server_name]
        
        logger.info(f"Creating new server instance for {server_name}")
        
        # Check if mcp_servers already has mcpServers key
        if "mcpServers" in self.mcp_servers:
            config = self.mcp_servers
        else:
            # Wrap in mcpServers if needed
            config = {"mcpServers": self.mcp_servers}
            
        # Use mcp_tool_transform module's function
        server_instance = await get_server_instance(server_name, config)
        if server_instance:
            self._server_instances[server_name] = server_instance
            logger.info(f"Created and stored server instance for {server_name}")
        else:
            logger.warning(f"Failed to create server instance for {server_name}")
        return server_instance

    async def call_tool(self, server_name: str, function_name: str, parameters: Dict[str, Any]) -> Any:
        """Call MCP tool"""
        from mas_arena.tools.mcp_tool_transform import call_mcp_tool
        
        try:
            logger.info(f"Calling tool {function_name} on server {server_name}")
            
            # Check if mcp_servers already has mcpServers key
            if "mcpServers" in self.mcp_servers:
                config = self.mcp_servers
            else:
                # Wrap in mcpServers if needed
                config = {"mcpServers": self.mcp_servers}
                
            # Use mcp_tool_transform module's function
            result = await call_mcp_tool(server_name, function_name, parameters, config)
            
            if "error" in result:
                logger.warning(f"Tool call error: {result['error']}")
            else:
                logger.info(f"Tool call successful: {server_name}.{function_name}")
                
            return result
        except Exception as e:
            logger.error(f"Error calling tool {function_name} on {server_name}: {e}")
            traceback.print_exc()
            return {"error": str(e)}

    def _get_mcp_tool_descriptions(self) -> List[Dict[str, Any]]:
        """Get MCP tool descriptions"""
        if not self.mcp_servers:
            logger.warning("No MCP servers configured")
            return []
        
        # Check if mcp_servers already has mcpServers key
        if "mcpServers" in self.mcp_servers:
            config = self.mcp_servers
            server_names = list(config["mcpServers"].keys())
        else:
            # Wrap in mcpServers if needed
            config = {"mcpServers": self.mcp_servers}
            server_names = list(self.mcp_servers.keys())
        
        # Use synchronous way to call asynchronous function
        import asyncio
        try:
            # Create a new event loop to run the async function
            logger.info(f"Getting tool descriptions for servers: {server_names}")
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            tool_descriptions = loop.run_until_complete(
                mcp_tool_desc_transform(server_names, config)
            )
            loop.close()
            
            # Log the result
            logger.info(f"Got {len(tool_descriptions)} tool descriptions")
            return tool_descriptions
        except Exception as e:
            logger.error(f"Error getting MCP tool descriptions: {e}")
            traceback.print_exc()
            return []

    @classmethod
    def from_config_file(cls, config_file_path: str, mock_mode: bool = False) -> "ToolManager":
        """从配置文件创建ToolManager实例"""
        try:
            config_path = Path(config_file_path)
            if not config_path.exists():
                logger.warning(f"Config file not found: {config_file_path}")
                return cls({}, mock_mode=True, tool_assignment_rules={})
            
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            mcp_servers = config.get("mcpServers", {})
            tool_assignment_rules = config.get("toolAssignmentRules", {})
            
            return cls(
                mcp_servers=mcp_servers,
                mock_mode=mock_mode,
                tool_assignment_rules=tool_assignment_rules,
                use_local_tools=True,
                use_mcp_tools=True
            )
        except Exception as e:
            logger.error(f"Error loading config file: {e}")
            traceback.print_exc()
            return cls({}, mock_mode=True, tool_assignment_rules={}) 