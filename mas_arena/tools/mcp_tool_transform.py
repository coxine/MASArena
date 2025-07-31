
import logging
import json
import traceback
import os
import asyncio
from typing import Dict, List, Any, Optional
import time
import importlib.util
import inspect
from pydantic.fields import FieldInfo

from mas_arena.mcp_collections.base import ActionCollection

logger = logging.getLogger(__name__)


def _get_pydantic_field_type(field: FieldInfo) -> str:
    """Gets the JSON schema type for a Pydantic field."""
    if issubclass(field.annotation, str):
        return "string"
    if issubclass(field.annotation, int):
        return "integer"
    if issubclass(field.annotation, float):
        return "number"
    if issubclass(field.annotation, bool):
        return "boolean"
    if issubclass(field.annotation, list):
        return "array"
    if issubclass(field.annotation, dict):
        return "object"
    return "string"


def _discover_tools_from_collections() -> Dict[str, Dict[str, Any]]:
    """
    Dynamically discover MCP tools from the mcp_collections subdirectories.

    Scans for ActionCollection subclasses and extracts public methods
    starting with "mcp_".

    Returns:
        A dictionary mapping tool name to a dictionary of its MCP functions
        and their introspected details (docstring, parameters).
    """
    tools_info = {}
    mcp_collections_dir = os.path.join(os.path.dirname(__file__), '..', 'mcp_collections')
    subdirs_to_scan = ['documents', 'intelligence', 'media', 'tools']

    for subdir in subdirs_to_scan:
        current_dir = os.path.join(mcp_collections_dir, subdir)
        if not os.path.isdir(current_dir):
            continue

        for filename in os.listdir(current_dir):
            if filename.endswith(".py") and not filename.startswith("__"):
                server_name = filename[:-3]
                module_path = os.path.join(current_dir, filename)
                
                try:
                    spec = importlib.util.spec_from_file_location(f"mas_arena.mcp_collections.{subdir}.{server_name}", module_path)
                    if spec and spec.loader:
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)
                        
                        for name, obj in inspect.getmembers(module):
                            if inspect.isclass(obj) and issubclass(obj, ActionCollection) and obj is not ActionCollection:
                                tool_name = getattr(obj, 'tool_name', None)
                                if not tool_name:
                                    continue

                                if tool_name not in tools_info:
                                    tools_info[tool_name] = {}
                                
                                for method_name, method_obj in inspect.getmembers(obj):
                                    if method_name.startswith("mcp_") and inspect.isfunction(method_obj):
                                        sig = inspect.signature(method_obj)
                                        docstring = inspect.getdoc(method_obj)
                                        
                                        parameters = {
                                            "type": "object",
                                            "properties": {},
                                            "required": []
                                        }
                                        
                                        for param_name, param in sig.parameters.items():
                                            if param_name == 'self':
                                                continue

                                            if isinstance(param.default, FieldInfo):
                                                field_info = param.default
                                                param_type = _get_pydantic_field_type(field_info)
                                                param_desc = field_info.description
                                                param_default = field_info.default
                                                
                                                parameters["properties"][param_name] = {
                                                    "type": param_type,
                                                    "description": param_desc
                                                }
                                                if param_default is not ... and param_default is not None:
                                                    parameters["properties"][param_name]["default"] = param_default
                                                else:
                                                    parameters["required"].append(param_name)

                                        tools_info[tool_name][method_name] = {
                                            "description": docstring,
                                            "parameters": parameters
                                        }

                except Exception as e:
                    logger.error(f"Error discovering tools in {filename}: {e}")

    return tools_info


async def mcp_tool_desc_transform(mcp_servers: List[str], mcp_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Transform MCP server configurations into tool descriptions
    
    Args:
        mcp_servers: List of MCP server names (tool_name)
        mcp_config: MCP server configuration dictionary
        
    Returns:
        List of tool descriptions
    """
    if not mcp_servers or not mcp_config or "mcpServers" not in mcp_config:
        logger.warning("MCP servers or config is empty")
        return []
        
    tool_descriptions = []
    
    try:
        all_tools = _discover_tools_from_collections()

        for server_name in mcp_servers:
            if server_name not in mcp_config["mcpServers"]:
                logger.warning(f"Server {server_name} not found in MCP config")
                continue

            if mcp_config["mcpServers"].get(server_name, {}).get("disabled", False):
                logger.info(f"Server {server_name} is disabled in config, skipping.")
                continue

            server_tools = all_tools.get(server_name)

            if not server_tools:
                logger.warning(f"No MCP functions found for server {server_name}")
                continue
            
            for function_name, tool_details in server_tools.items():
                tool_desc = {
                    "name": server_name,
                    "description": tool_details.get("description", f"Function {function_name} from {server_name} server."),
                    "server_name": server_name,
                    "function_name": function_name,
                    "parameters": tool_details.get("parameters", {"type": "object", "properties": {}}),
                }
                tool_descriptions.append(tool_desc)
            
    except Exception as e:
        logger.error(f"Error transforming MCP tool descriptions: {e}")
        traceback.print_exc()
    
    return tool_descriptions


async def get_server_instance(server_name: str, mcp_config: Dict[str, Any]) -> Optional[Any]:
    """
    Get or create MCP server instance
    
    Args:
        server_name: Server name
        mcp_config: MCP configuration
        
    Returns:
        Server instance
    """
    if not mcp_config or "mcpServers" not in mcp_config or server_name not in mcp_config["mcpServers"]:
        logger.warning(f"Server {server_name} not found in MCP config")
        return None
        
    try:
        # Get server configuration
        server_config = mcp_config["mcpServers"][server_name]
        
        # Skip disabled servers
        if server_config.get("disabled", False):
            logger.info(f"Server {server_name} is disabled, skipping")
            return None
            
        # API type servers don't need persistent connections
        if server_config.get("type", "") == "api":
            logger.info(f"API server {server_name} doesn't need persistent connection")
            return None
            
        # Create server process
        command = server_config.get("command", "")
        args = server_config.get("args", [])
        env = server_config.get("env", {})
        cwd = server_config.get("cwd")
        timeout = server_config.get("timeout", 60.0)
        
        # Process environment variables placeholders
        processed_env = {}
        for key, value in env.items():
            if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                env_var_name = value[2:-1]
                env_value = os.environ.get(env_var_name)
                if env_value is None:
                    logger.warning(f"Environment variable {env_var_name} not found for server {server_name}")
                    processed_env[key] = ""
                else:
                    processed_env[key] = env_value
            else:
                processed_env[key] = value
        
        # Merge environment variables
        full_env = os.environ.copy()
        full_env.update(processed_env)
        
        # Build full command
        full_command = [command] + args
        
        logger.info(f"Starting server {server_name} with command: {full_command}")
        
        # Create server process
        try:
            process = await asyncio.create_subprocess_exec(
                *full_command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=full_env,
                cwd=cwd
            )
            
            # Check if process started successfully
            if process.returncode is not None:
                logger.error(f"Failed to start server {server_name}: process exited with code {process.returncode}")
                return None
                
            # Wait a bit to make sure process starts properly
            await asyncio.sleep(0.5)
            
            # Check again if process is still running
            if process.returncode is not None:
                stderr_data = await process.stderr.read()
                logger.error(f"Server {server_name} exited prematurely with code {process.returncode}: {stderr_data.decode('utf-8', errors='replace')}")
                return None
                
            # Create server instance object
            server_instance = {
                "name": server_name,
                "process": process,
                "config": server_config,
                "type": server_config.get("type", "stdio"),
                "start_time": time.time()
            }
            
            # Add timeout from client_session_timeout_seconds if available
            if "client_session_timeout_seconds" in server_config:
                server_instance["timeout"] = server_config["client_session_timeout_seconds"]
            elif "timeout" in server_config:
                server_instance["timeout"] = server_config["timeout"]
            else:
                server_instance["timeout"] = 60.0
            
            logger.info(f"Created server instance for {server_name}")
            return server_instance
            
        except FileNotFoundError:
            logger.error(f"Command not found: {command}")
            return None
        except PermissionError:
            logger.error(f"Permission denied when executing: {command}")
            return None
        
    except Exception as e:
        logger.error(f"Error creating server instance for {server_name}: {e}")
        traceback.print_exc()
        return None

async def cleanup_server(server: Any) -> None:
    """
    Clean up server instance
    
    Args:
        server: Server instance
    """
    if not server:
        return
        
    try:
        server_name = server.get("name", "unknown")
        logger.info(f"Cleaning up server {server_name}")
        
        # Get process
        process = server.get("process")
        if not process:
            logger.warning(f"No process found for server {server_name}")
            return
            
        # Check if process is still running
        if process.returncode is not None:
            logger.info(f"Process for server {server_name} already terminated with code {process.returncode}")
            return
            
        # Try graceful termination first
        try:
            logger.info(f"Sending terminate signal to server {server_name}")
            process.terminate()
            
            # Wait for process to terminate with timeout
            try:
                await asyncio.wait_for(process.wait(), timeout=5.0)
                logger.info(f"Server {server_name} terminated gracefully")
            except asyncio.TimeoutError:
                # Force kill if process doesn't terminate within timeout
                logger.warning(f"Server {server_name} did not terminate gracefully, forcing kill")
                process.kill()
                await process.wait()
                logger.info(f"Server {server_name} killed forcefully")
        except ProcessLookupError:
            logger.info(f"Process for server {server_name} already gone")
        
        # Close pipes
        if process.stdin:
            if not process.stdin.is_closing():
                process.stdin.close()
            await process.stdin.wait_closed()
            
        if process.stdout:
            if not process.stdout.is_closing():
                process.stdout.close()
                
        if process.stderr:
            if not process.stderr.is_closing():
                process.stderr.close()
                
        # Log cleanup duration
        if "start_time" in server:
            duration = time.time() - server["start_time"]
            logger.info(f"Server {server_name} ran for {duration:.2f} seconds")
            
        logger.info(f"Cleaned up server {server_name}")
    except Exception as e:
        logger.error(f"Error cleaning up server {server.get('name', 'unknown')}: {e}")
        traceback.print_exc()
    
async def call_api(server_name: str, function_name: str, parameters: Dict[str, Any], mcp_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Call API type MCP tool
    
    Args:
        server_name: Server name
        function_name: Function name to call
        parameters: Tool parameters
        mcp_config: MCP configuration
        
    Returns:
        Tool call result
    """
    # For API type servers, send HTTP request directly
    try:
        import requests
        
        if server_name not in mcp_config.get("mcpServers", {}):
            logger.error(f"Server {server_name} not found in MCP config")
            return {"error": f"Server {server_name} not found in MCP config"}
            
        server_config = mcp_config["mcpServers"][server_name]
        base_url = server_config.get("url", "")
        
        if not base_url:
            logger.error(f"No URL configured for API server {server_name}")
            return {"error": f"No URL configured for API server {server_name}"}
            
        # Build API request
        url = f"{base_url}/{function_name}"
        headers = server_config.get("headers", {})
        timeout = server_config.get("timeout", 60.0)
        
        # Add function_name to parameters for semantic consistency
        if isinstance(parameters, dict):
            parameters = parameters.copy()  # Create a copy to avoid modifying the original
            parameters["function_name"] = function_name
        
        logger.info(f"Calling API function {function_name} on {server_name} at {url}")
        
        # Process environment variables in headers
        processed_headers = {}
        for key, value in headers.items():
            if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                env_var_name = value[2:-1]
                processed_headers[key] = os.environ.get(env_var_name, "")
            else:
                processed_headers[key] = value
        
        # Send request with timeout
        response = requests.post(
            url, 
            json=parameters, 
            headers=processed_headers,
            timeout=timeout
        )
        
        if response.status_code != 200:
            error_msg = f"API call failed with status code {response.status_code}: {response.text[:100]}..."
            logger.error(error_msg)
            return {"error": error_msg}
            
        try:
            result = response.json()
            logger.info(f"Received successful response from {server_name}.{function_name}")
            return {"result": result}
        except json.JSONDecodeError as e:
            error_msg = f"Failed to parse JSON response: {e}"
            logger.error(f"{error_msg}. Response text: {response.text[:100]}...")
            return {"error": error_msg}
            
    except requests.exceptions.Timeout:
        error_msg = f"API call to {server_name}.{function_name} timed out"
        logger.error(error_msg)
        return {"error": error_msg}
    except requests.exceptions.ConnectionError as e:
        error_msg = f"Connection error calling {server_name}.{function_name}: {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg}
    except Exception as e:
        logger.error(f"Error calling API {function_name} on {server_name}: {e}")
        traceback.print_exc()
        return {"error": str(e)}
    
async def call_function_tool(server_name: str, function_name: str, parameters: Dict[str, Any], mcp_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Call function type MCP tool
    
    Args:
        server_name: Server name
        function_name: Function name to call
        parameters: Tool parameters
        mcp_config: MCP configuration
        
    Returns:
        Tool call result
    """
    # For function type servers, import and call the function directly
    try:
        if server_name not in mcp_config.get("mcpServers", {}):
            logger.error(f"Server {server_name} not found in MCP config")
            return {"error": f"Server {server_name} not found in MCP config"}
            
        server_config = mcp_config["mcpServers"][server_name]
        module_path = server_config.get("module_path", "")
        
        if not module_path:
            logger.error(f"No module path configured for function tool server {server_name}")
            return {"error": f"No module path configured for function tool server {server_name}"}
            
        logger.info(f"Calling function tool {function_name} in module {module_path}")
        
        # Import module
        module_parts = module_path.split(".")
        module_name = ".".join(module_parts)
        
        try:
            # Dynamic import with error handling
            try:
                module = __import__(module_name, fromlist=[function_name])
            except ImportError as e:
                logger.error(f"Failed to import module {module_name}: {e}")
                return {"error": f"Could not import module {module_name}: {str(e)}"}
                
            # Get function with error handling
            try:
                function = getattr(module, function_name)
            except AttributeError as e:
                logger.error(f"Function {function_name} not found in module {module_name}: {e}")
                return {"error": f"Function {function_name} not found in module {module_name}"}
            
            # Validate function is callable
            if not callable(function):
                logger.error(f"{function_name} in module {module_name} is not callable")
                return {"error": f"{function_name} in module {module_name} is not callable"}
            
            # Call function with timeout
            try:
                # Prepare arguments - handle both positional and keyword args
                if isinstance(parameters, dict):
                    # Execute function with timeout
                    result = await asyncio.wait_for(
                        asyncio.to_thread(function, **parameters),
                        timeout=server_config.get("timeout", 60.0)
                    )
                else:
                    logger.error(f"Invalid parameters type for {function_name}: expected dict, got {type(parameters)}")
                    return {"error": f"Invalid parameters type: expected dict, got {type(parameters)}"}
                
                logger.info(f"Successfully called function {function_name} in {module_name}")
                
                # Process result
                if result is None:
                    return {"result": "Function executed successfully but returned None"}
                    
                # Try to make result JSON serializable
                try:
                    # Test JSON serialization
                    json.dumps(result)
                    return {"result": result}
                except (TypeError, OverflowError) as e:
                    # If result is not JSON serializable, convert to string
                    logger.warning(f"Function result not JSON serializable: {e}. Converting to string.")
                    return {"result": str(result)}
                    
            except asyncio.TimeoutError:
                logger.error(f"Function call to {function_name} timed out")
                return {"error": f"Function call to {function_name} timed out"}
            except Exception as e:
                logger.error(f"Error executing function {function_name}: {e}")
                traceback.print_exc()
                return {"error": f"Error executing function: {str(e)}"}
                
        except Exception as e:
            logger.error(f"Unexpected error calling function {function_name}: {e}")
            traceback.print_exc()
            return {"error": f"Unexpected error: {str(e)}"}
            
    except Exception as e:
        logger.error(f"Error calling function {function_name} on {server_name}: {e}")
        traceback.print_exc()
        return {"error": str(e)}

async def call_mcp_tool(server_name: str, function_name: str, parameters: Dict[str, Any], mcp_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Call MCP tool, choosing appropriate call method based on server type
    
    Args:
        server_name: Server name
        function_name: Function name to call
        parameters: Tool parameters
        mcp_config: MCP configuration
        
    Returns:
        Tool call result
    """
    if not mcp_config or "mcpServers" not in mcp_config or server_name not in mcp_config["mcpServers"]:
        logger.error(f"Server {server_name} not found in MCP config")
        return {"error": f"Server {server_name} not found in MCP config"}
        
    server_config = mcp_config["mcpServers"][server_name]
    server_type = server_config.get("type", "stdio")
    
    logger.info(f"Calling function {function_name} on server {server_name} with type {server_type}")
    
    try:
        # Choose call method based on server type
        if server_type == "api":
            return await call_api(server_name, function_name, parameters, mcp_config)
        elif server_type == "function_tool":
            return await call_function_tool(server_name, function_name, parameters, mcp_config)
        else:
            # For stdio type servers, start process and communicate via stdin/stdout
            server_instance = await get_server_instance(server_name, mcp_config)
            if not server_instance:
                logger.error(f"Failed to create server instance for {server_name}")
                return {"error": f"Failed to create server instance for {server_name}"}
                
            try:
                process = server_instance.get("process")
                if not process:
                    logger.error(f"No process found for server {server_name}")
                    return {"error": f"No process found for server {server_name}"}
                
                # Prepare input data
                input_data = {
                    "function_name": function_name,
                    "name": function_name,  # Keep name for backward compatibility
                    "arguments": parameters
                }
                
                # Send input data
                input_json = json.dumps(input_data)
                logger.debug(f"Sending to {server_name}: {input_json}")
                input_bytes = (input_json + "\n").encode()
                process.stdin.write(input_bytes)
                await process.stdin.drain()
                
                # Read output data with timeout
                try:
                    # Get timeout from server instance or config
                    timeout = server_instance.get("timeout", server_config.get("timeout", 60.0))
                    logger.info(f"Setting timeout for {server_name} to {timeout} seconds")
                    
                    # Set a timeout for reading response
                    output_line = await asyncio.wait_for(
                        process.stdout.readline(),
                        timeout=timeout
                    )
                    output_text = output_line.decode().strip()
                    logger.debug(f"Received from {server_name}: {output_text[:200]}...")
                    
                    # Check if output is empty
                    if not output_text:
                        logger.error(f"Empty response from {server_name}")
                        
                        # Try to read stderr for error information
                        stderr_data = await asyncio.wait_for(
                            process.stderr.read(1024),  # Read up to 1KB of stderr
                            timeout=1.0
                        )
                        stderr_text = stderr_data.decode(errors='replace') if stderr_data else ""
                        
                        if stderr_text:
                            logger.error(f"Stderr from {server_name}: {stderr_text}")
                            return {"error": f"Empty response from {server_name}. Stderr: {stderr_text[:200]}..."}
                        else:
                            return {"error": f"Empty response from {server_name}"}
                    
                    try:
                        output_data = json.loads(output_text)
                        logger.info(f"Received response from {server_name}.{function_name}")
                        
                        # Check for error in response
                        if isinstance(output_data, dict) and "error" in output_data:
                            logger.warning(f"Error in tool response: {output_data['error']}")
                            return {"error": output_data["error"]}
                        
                        return {"result": output_data}
                    except json.JSONDecodeError as e:
                        logger.error(f"Error parsing output from {server_name}: {e}")
                        logger.error(f"Raw output: {output_text[:500]}")
                        
                        # Try to extract content if it looks like debug output with JSON
                        json_start = output_text.find('{')
                        json_end = output_text.rfind('}')
                        
                        if json_start >= 0 and json_end > json_start:
                            # Try to extract JSON from the output
                            try:
                                json_part = output_text[json_start:json_end+1]
                                extracted_data = json.loads(json_part)
                                logger.info(f"Successfully extracted JSON from output")
                                return {"result": extracted_data}
                            except json.JSONDecodeError:
                                pass
                        
                        return {"error": f"Failed to parse output: {output_text[:200]}..."}
                except asyncio.TimeoutError:
                    logger.error(f"Timeout waiting for response from {server_name}")
                    return {"error": f"Timeout waiting for response from {server_name}"}
                    
            except Exception as e:
                logger.error(f"Error calling function {function_name} on {server_name}: {e}")
                traceback.print_exc()
                return {"error": str(e)}
    except Exception as e:
        logger.error(f"Unexpected error calling function {function_name} on {server_name}: {e}")
        traceback.print_exc()
        return {"error": f"Unexpected error: {str(e)}"}
