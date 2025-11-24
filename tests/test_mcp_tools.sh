#!/bin/bash
#
# MCP Browser Use Server Test Suite
# Tests all MCP tools via Python MCP client over SSE
#
# Usage: ./test_mcp_tools.sh
#

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# MCP Server URL
MCP_SERVER_URL="${MCP_SERVER_URL:-http://127.0.0.1:8081/sse}"

# Function to print section headers
print_section() {
    echo ""
    echo -e "${BOLD}${CYAN}================================================================================${NC}"
    echo -e "${BOLD}${CYAN}$1${NC}"
    echo -e "${BOLD}${CYAN}================================================================================${NC}"
    echo ""
}

# Function to print success
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

# Function to print error
print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# Function to print info
print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

print_section "MCP BROWSER USE SERVER - CLI TEST SUITE"
print_info "MCP Server URL: $MCP_SERVER_URL"

# Check if server is running
print_section "PRE-FLIGHT CHECK"
print_info "Checking if MCP server is accessible..."
if curl -s -I "$MCP_SERVER_URL" 2>&1 | grep -q "HTTP/1.1 200"; then
    print_success "MCP server is running and accessible"
else
    print_error "Cannot connect to MCP server at $MCP_SERVER_URL"
    echo "Please ensure the Docker container is running: docker-compose up -d"
    exit 1
fi

# Create Python test script inline
print_section "EXECUTING MCP TESTS"
print_info "Creating temporary Python test client..."

cat > /tmp/mcp_cli_test.py << 'PYTHON_EOF'
import asyncio
import json
import sys
import time
from typing import Any, Dict
from mcp import ClientSession
from mcp.client.sse import sse_client


class Colors:
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    CYAN = '\033[0;36m'
    BOLD = '\033[1m'
    NC = '\033[0m'


def print_success(msg):
    print(f"{Colors.GREEN}✓ {msg}{Colors.NC}")


def print_error(msg):
    print(f"{Colors.RED}✗ {msg}{Colors.NC}")


def print_info(msg):
    print(f"{Colors.BLUE}ℹ {msg}{Colors.NC}")


def print_section(msg):
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'=' * 80}{Colors.NC}")
    print(f"{Colors.BOLD}{Colors.CYAN}{msg}{Colors.NC}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'=' * 80}{Colors.NC}\n")


class MCPTestRunner:
    def __init__(self, server_url: str):
        self.server_url = server_url
        self.session = None
        self.total_tests = 0
        self.passed_tests = 0
        self.failed_tests = 0
        
    def record_test(self, passed: bool):
        self.total_tests += 1
        if passed:
            self.passed_tests += 1
        else:
            self.failed_tests += 1
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Call an MCP tool and return the result."""
        try:
            result = await self.session.call_tool(tool_name, arguments=arguments)
            return result
        except Exception as e:
            print_error(f"Error calling tool {tool_name}: {e}")
            return None
    
    async def test_list_tools(self):
        """Test 1: List all available tools"""
        print_section("TEST 1: List Available Tools")
        print_info("Fetching list of available MCP tools...")
        
        try:
            tools = await self.session.list_tools()
            print_success(f"Successfully retrieved {len(tools.tools)} tools")
            for tool in tools.tools[:5]:  # Show first 5
                print(f"  - {tool.name}: {tool.description[:60]}...")
            if len(tools.tools) > 5:
                print(f"  ... and {len(tools.tools) - 5} more")
            self.record_test(True)
            return True
        except Exception as e:
            print_error(f"Failed to retrieve tools list: {e}")
            self.record_test(False)
            return False
    
    async def test_browser_use(self):
        """Test 2: Browser Use (Async Task)"""
        print_section("TEST 2: Browser Use Tool (Async Task)")
        print_info("Creating async browser task to navigate to quotes.toscrape.com...")
        
        try:
            result = await self.call_tool("browser_use", {
                "url": "https://quotes.toscrape.com/",
                "action": "Navigate to the page and count how many quotes are visible"
            })
            
            if result and len(result.content) > 0:
                content = json.loads(result.content[0].text)
                task_id = content.get("task_id")
                if task_id:
                    print_success(f"Task created with ID: {task_id}")
                    self.record_test(True)
                    return task_id
            
            print_error("Failed to create browser task")
            self.record_test(False)
            return None
        except Exception as e:
            print_error(f"Error in browser_use: {e}")
            self.record_test(False)
            return None
    
    async def test_browser_get_result(self, task_id: str):
        """Test 3: Get Task Result (Polling)"""
        print_section("TEST 3: Browser Get Result (Polling)")
        print_info(f"Polling for task result (max 60 seconds)...")
        
        for i in range(12):
            await asyncio.sleep(5)
            print_info(f"Attempt {i+1}: Checking task status...")
            
            try:
                result = await self.call_tool("browser_get_result", {"task_id": task_id})
                if result and len(result.content) > 0:
                    content = json.loads(result.content[0].text)
                    status = content.get("status")
                    
                    if status == "completed":
                        print_success("Task completed successfully!")
                        print(json.dumps(content, indent=2)[:500])
                        self.record_test(True)
                        return True
                    elif status == "failed":
                        print_error("Task failed")
                        print(json.dumps(content, indent=2))
                        self.record_test(False)
                        return False
            except Exception as e:
                print_error(f"Error checking task status: {e}")
        
        print_error("Task timeout after 60 seconds")
        self.record_test(False)
        return False
    
    async def test_browser_navigate(self):
        """Test 4: Browser Navigate (Session)"""
        print_section("TEST 4: Browser Navigate Tool")
        print_info("Creating session and navigating to example.com...")
        
        try:
            result = await self.call_tool("browser_navigate", {"url": "https://example.com"})
            if result and len(result.content) > 0:
                content = json.loads(result.content[0].text)
                session_id = content.get("session_id")
                if session_id:
                    print_success(f"Session created with ID: {session_id}")
                    self.record_test(True)
                    return session_id
            
            print_error("Failed to create session")
            self.record_test(False)
            return None
        except Exception as e:
            print_error(f"Error in browser_navigate: {e}")
            self.record_test(False)
            return None
    
    async def test_browser_get_state(self, session_id: str):
        """Test 5: Browser Get State"""
        print_section("TEST 5: Browser Get State Tool")
        print_info(f"Getting current state of session {session_id}...")
        
        try:
            result = await self.call_tool("browser_get_state", {
                "session_id": session_id,
                "screenshot": False
            })
            if result and len(result.content) > 0:
                content = json.loads(result.content[0].text)
                if "url" in content:
                    print_success("Successfully retrieved browser state")
                    print(json.dumps(content, indent=2)[:400])
                    self.record_test(True)
                    return True
            
            print_error("Failed to retrieve browser state")
            self.record_test(False)
            return False
        except Exception as e:
            print_error(f"Error in browser_get_state: {e}")
            self.record_test(False)
            return False
    
    async def test_browser_list_sessions(self):
        """Test 6: Browser List Sessions"""
        print_section("TEST 6: Browser List Sessions Tool")
        print_info("Listing all active sessions...")
        
        try:
            result = await self.call_tool("browser_list_sessions", {})
            if result and len(result.content) > 0:
                content = json.loads(result.content[0].text)
                print_success("Successfully retrieved sessions list")
                print(json.dumps(content, indent=2))
                self.record_test(True)
                return True
            
            print_error("Failed to retrieve sessions list")
            self.record_test(False)
            return False
        except Exception as e:
            print_error(f"Error in browser_list_sessions: {e}")
            self.record_test(False)
            return False
    
    async def test_browser_list_tabs(self, session_id: str):
        """Test 7: Browser List Tabs"""
        print_section("TEST 7: Browser List Tabs Tool")
        print_info(f"Listing tabs in session {session_id}...")
        
        try:
            result = await self.call_tool("browser_list_tabs", {"session_id": session_id})
            if result and len(result.content) > 0:
                content = json.loads(result.content[0].text)
                print_success("Successfully retrieved tabs list")
                print(json.dumps(content, indent=2))
                self.record_test(True)
                return True
            
            print_error("Failed to retrieve tabs list")
            self.record_test(False)
            return False
        except Exception as e:
            print_error(f"Error in browser_list_tabs: {e}")
            self.record_test(False)
            return False
    
    async def test_browser_extract_content(self, session_id: str):
        """Test 8: Browser Extract Content"""
        print_section("TEST 8: Browser Extract Content Tool")
        print_info(f"Extracting content from session {session_id}...")
        
        try:
            result = await self.call_tool("browser_extract_content", {
                "session_id": session_id,
                "instruction": "Extract the main heading text"
            })
            if result and len(result.content) > 0:
                content = json.loads(result.content[0].text)
                print_success("Successfully extracted content")
                print(json.dumps(content, indent=2)[:400])
                self.record_test(True)
                return True
            
            print_error("Failed to extract content")
            self.record_test(False)
            return False
        except Exception as e:
            print_error(f"Error in browser_extract_content: {e}")
            self.record_test(False)
            return False
    
    async def test_browser_close_session(self, session_id: str):
        """Test 9: Browser Close Session"""
        print_section("TEST 9: Browser Close Session Tool")
        print_info(f"Closing session {session_id}...")
        
        try:
            result = await self.call_tool("browser_close_session", {"session_id": session_id})
            if result and len(result.content) > 0:
                content = json.loads(result.content[0].text)
                print_success("Successfully closed session")
                print(json.dumps(content, indent=2))
                self.record_test(True)
                return True
            
            print_error("Failed to close session")
            self.record_test(False)
            return False
        except Exception as e:
            print_error(f"Error in browser_close_session: {e}")
            self.record_test(False)
            return False
    
    async def test_hubertusbecker_summary(self):
        """Test 10: FINAL TEST - Summarize hubertusbecker.com"""
        print_section("TEST 10: FINAL TEST - Summarize hubertusbecker.com")
        print_info("Creating task to summarize hubertusbecker.com...")
        
        try:
            result = await self.call_tool("browser_use", {
                "url": "https://hubertusbecker.com",
                "action": "Summarize the main content and purpose of this website"
            })
            
            if result and len(result.content) > 0:
                content = json.loads(result.content[0].text)
                
                # Check if this is a synchronous result (has final_result)
                if "final_result" in content:
                    print_success("Summary completed synchronously!")
                    print(f"\n{Colors.BOLD}{Colors.GREEN}SUMMARY OF HUBERTUSBECKER.COM:{Colors.NC}")
                    print(json.dumps(content, indent=2))
                    self.record_test(True)
                    return True
                
                # Check if this is an async task (has task_id)
                task_id = content.get("task_id")
                if task_id:
                    print_success(f"Summary task created with ID: {task_id}")
                    print_info("Waiting for summary to complete...")
                    
                    for i in range(24):
                        await asyncio.sleep(5)
                        print_info(f"Attempt {i+1}: Checking summary status...")
                        
                        result = await self.call_tool("browser_get_result", {"task_id": task_id})
                        if result and len(result.content) > 0:
                            content = json.loads(result.content[0].text)
                            status = content.get("status")
                            
                            if status == "completed":
                                print_success("Summary completed successfully!")
                                print(f"\n{Colors.BOLD}{Colors.GREEN}SUMMARY OF HUBERTUSBECKER.COM:{Colors.NC}")
                                print(json.dumps(content, indent=2))
                                self.record_test(True)
                                return True
                            elif status == "failed":
                                print_error("Summary task failed")
                                print(json.dumps(content, indent=2))
                                self.record_test(False)
                                return False
                    
                    print_error("Summary task timeout after 120 seconds")
                    self.record_test(False)
                    return False
            
            print_error("Failed to create summary task")
            self.record_test(False)
            return False
        except Exception as e:
            print_error(f"Error in hubertusbecker summary: {e}")
            import traceback
            traceback.print_exc()
            self.record_test(False)
            return False
    
    def print_summary(self):
        """Print final test summary"""
        print_section("TEST SUMMARY")
        print(f"{Colors.BOLD}Total Tests:{Colors.NC} {self.total_tests}")
        print(f"{Colors.GREEN}{Colors.BOLD}Passed:{Colors.NC} {self.passed_tests}")
        print(f"{Colors.RED}{Colors.BOLD}Failed:{Colors.NC} {self.failed_tests}")
        
        if self.total_tests > 0:
            pass_rate = (self.passed_tests * 100) // self.total_tests
            print(f"{Colors.BOLD}Pass Rate:{Colors.NC} {pass_rate}%")
        
        print()
        if self.failed_tests == 0:
            print(f"{Colors.GREEN}{Colors.BOLD}✓ ALL TESTS PASSED!{Colors.NC}")
            return 0
        else:
            print(f"{Colors.RED}{Colors.BOLD}✗ SOME TESTS FAILED{Colors.NC}")
            return 1
    
    async def run_all_tests(self):
        """Run all tests in sequence"""
        async with sse_client(self.server_url) as (read, write):
            async with ClientSession(read, write) as session:
                self.session = session
                
                # Initialize
                await session.initialize()
                
                # Test 1: List tools
                await self.test_list_tools()
                
                # Test 2-3: Browser use and get result
                task_id = await self.test_browser_use()
                if task_id:
                    await self.test_browser_get_result(task_id)
                else:
                    self.record_test(False)  # Mark get_result as failed too
                
                # Test 4-9: Session tests
                session_id = await self.test_browser_navigate()
                if session_id:
                    await self.test_browser_get_state(session_id)
                    await self.test_browser_list_sessions()
                    await self.test_browser_list_tabs(session_id)
                    await self.test_browser_extract_content(session_id)
                    await self.test_browser_close_session(session_id)
                else:
                    # Mark skipped tests as failed
                    for _ in range(5):
                        self.record_test(False)
                
                # Test 10: Final test - hubertusbecker.com summary
                await self.test_hubertusbecker_summary()
                
                # Print summary
                return self.print_summary()


async def main():
    server_url = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8081/sse"
    runner = MCPTestRunner(server_url)
    exit_code = await runner.run_all_tests()
    sys.exit(exit_code)


if __name__ == "__main__":
    asyncio.run(main())
PYTHON_EOF

# Run the Python test script
print_info "Running Python MCP test client..."
uv run python /tmp/mcp_cli_test.py "$MCP_SERVER_URL"
exit_code=$?

# Cleanup
rm -f /tmp/mcp_cli_test.py

exit $exit_code
