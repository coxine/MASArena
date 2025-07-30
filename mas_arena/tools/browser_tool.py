# coding: utf-8
import base64
import subprocess
from pathlib import Path
from typing import List
import sys
import importlib
import asyncio

from langchain.tools import StructuredTool

from mas_arena.tools.base import ToolFactory

BROWSER = "browser"

def import_and_install(package_name: str):
    """Tries to import a package, and if it fails, attempts to install it and then import it again."""
    try:
        return __import__(package_name)
    except ImportError:
        print(f"Package '{package_name}' not found. Attempting to install...")
        try:
            # Use subprocess.run to capture both stdout and stderr
            process = subprocess.run(
                [sys.executable, "-m", "pip", "install", package_name],
                capture_output=True,
                text=True,
                check=True
            )
            print(f"Successfully installed '{package_name}'.")
            importlib.invalidate_caches()
            return __import__(package_name)
        except subprocess.CalledProcessError as e:
            # Now we can inspect both e.stdout and e.stderr
            stdout_details = e.stdout.strip() if e.stdout else "No stdout output."
            stderr_details = e.stderr.strip() if e.stderr else "No stderr output."
            print(f"--- pip install failed ---")
            print(f"Failed to install '{package_name}' via pip. Error details below:")
            print("--- STDOUT ---")
            print(stdout_details)
            print("--- STDERR ---")
            print(stderr_details)
            print(f"--- End of pip error ---")
            raise ImportError(f"Could not install {package_name}") from e
        except Exception as e:
            print(f"An unexpected error occurred during the installation of '{package_name}': {e}")
            raise ImportError(f"Could not import or install {package_name}") from e

class Browser:
    def __init__(self, **kwargs) -> None:
        self.initialized = False
        
        # Ensure playwright is available
        import_and_install("playwright")

        # Ensure browser binaries are installed
        print("Checking/installing Playwright browser binaries...")
        try:
            # Using sys.executable to ensure we use the correct playwright
            subprocess.check_call([sys.executable, "-m", "playwright", "install"], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        except Exception as e:
            print(f"Warning: Failed to install playwright browsers, but continuing. Error: {e}")

        self._finish = False
        self.record_trace = kwargs.get("enable_recording", False)
        self.sleep_after_init = kwargs.get("sleep_after_init", False)

        # Initialize async resources
        self.context_manager = None
        self.playwright = None
        self.browser = None
        self.context = None
        self.page = None

    async def init(self) -> None:
        from playwright.async_api import async_playwright

        if self.initialized:
            return

        self.context_manager = async_playwright()
        self.playwright = await self.context_manager.start()
        self.browser = await self._create_browser()
        self.context = await self._create_browser_context()

        if self.record_trace:
            await self.context.tracing.start(screenshots=True, snapshots=True)

        self.page = await self.context.new_page()
        self.initialized = True

    async def _create_browser(self):
        browse_name = "chromium"
        browse = getattr(self.playwright, browse_name)
        headless = True
        slow_mo = 0
        disable_security_args = ['--disable-web-security', '--disable-site-isolation-trials', '--disable-features=IsolateOrigins,site-per-process']
        args = ['--no-sandbox', '--disable-crash-reporter', '--disable-blink-features=AutomationControlled', '--disable-infobars', '--disable-background-timer-throttling', '--disable-popup-blocking', '--disable-backgrounding-occluded-windows', '--disable-renderer-backgrounding', '--disable-window-activation', '--disable-focus-on-load', '--no-first-run', '--no-default-browser-check', '--no-startup-window', '--window-position=0,0', '--window-size=1280,720'] + disable_security_args
        browser = await browse.launch(
            headless=headless,
            slow_mo=slow_mo,
            args=args,
        )
        return browser

    async def _create_browser_context(self):
        from playwright.async_api import ViewportSize

        if not self.browser:
            raise RuntimeError("Browser not initialized")
            
        viewport_size = ViewportSize(width=1280, height=720)
        disable_security = True

        context = await self.browser.new_context(viewport=viewport_size,
                                      no_viewport=False,
                                      java_script_enabled=True,
                                      bypass_csp=disable_security,
                                      ignore_https_errors=disable_security,
                                      device_scale_factor=1)
        return context

    async def navigate(self, url: str) -> str:
        """Navigate to a URL."""
        if not self.page:
            return "Browser not initialized"
        try:
            await self.page.goto(url)
            return f"Navigated to {url}"
        except Exception as e:
            return f"Failed to navigate to {url}: {e}"

    async def get_page_content(self, clean=True) -> str:
        """
        Get the text content of the current page.
        Args:
            clean: Whether to run a cleaning script to remove irrelevant content.
        """
        if not self.page:
            return "Browser not initialized"
        try:
            if clean:
                # A simple script to remove common clutter like nav, footer, scripts, styles
                js_script = """() => {
                    const doc = document.cloneNode(true);
                    doc.querySelectorAll('nav, footer, script, style, aside, [role="navigation"], [role="banner"], [role="contentinfo"]').forEach(el => el.remove());
                    return doc.body.innerText;
                }"""
                return await self.page.evaluate(js_script)
            else:
                return await self.page.inner_text('body')
        except Exception as e:
            return f"Failed to get page content: {e}"

    def get_current_url(self) -> str:
        """Get the current URL."""
        if not self.page:
            return "Browser not initialized"
        return self.page.url

    async def screenshot(self, full_page: bool = False) -> str:
        """Returns a base64 encoded screenshot of the current page."""
        if not self.page:
            return "Browser not initialized"
        try:
            await self.page.bring_to_front()
            await self.page.wait_for_load_state(timeout=2000)
        except:
            pass

        screenshot = await self.page.screenshot(
            full_page=full_page,
            animations='disabled',
            timeout=600000
        )
        screenshot_base64 = base64.b64encode(screenshot).decode('utf-8')
        return screenshot_base64

    async def close(self) -> None:
        if not self.initialized:
            return
        if self.record_trace and self.context:
            await self.save_trace("trace.zip")

        if self.page:
            await self.page.close()
        if self.context:
            await self.context.close()
        if self.browser:
            await self.browser.close()
        if hasattr(self, 'context_manager') and self.context_manager:
            await self.context_manager.__aexit__(None, None, None)
        self.initialized = False

    async def save_trace(self, trace_path: str | Path) -> None:
        if self.context and hasattr(self.context, 'tracing'):
            await self.context.tracing.stop(path=trace_path)


def run_async(async_func):
    """运行异步函数的同步包装器"""
    def wrapper(*args, **kwargs):
        try:
            loop = asyncio.get_running_loop()
            # 如果已经在事件循环中，使用 asyncio.create_task 而不是 run_in_executor
            import nest_asyncio
            nest_asyncio.apply()
            return asyncio.run(async_func(*args, **kwargs))
        except RuntimeError:
            # 没有运行中的事件循环，创建一个新的
            return asyncio.run(async_func(*args, **kwargs))
    return wrapper


@ToolFactory.register(name=BROWSER, desc="A tool for browsing the web.")
class BrowserTool:
    def __init__(self):
        self.browser = None
        self._initialized = False
        self._cleanup_registered = False
        try:
            self.browser = Browser()
            # 注册清理函数，在程序退出时执行
            import atexit
            if not self._cleanup_registered:
                atexit.register(self._cleanup_on_exit)
                self._cleanup_registered = True
        except Exception as e:
            print(f"Error: tool browser load failed - {e}")
            raise

    def _cleanup_on_exit(self):
        """程序退出时的清理函数"""
        if self._initialized and self.browser:
            try:
                # 使用同步方式强制关闭
                if hasattr(self.browser, 'browser') and self.browser.browser:
                    # 直接关闭浏览器进程，不等待异步操作
                    import subprocess
                    import psutil
                    try:
                        # 尝试优雅关闭
                        asyncio.run(self.browser.close())
                    except:
                        # 如果优雅关闭失败，强制终止相关进程
                        pass
                self._initialized = False
            except:
                pass

    def _ensure_initialized(self):
        """确保浏览器已初始化"""
        if not self._initialized and self.browser:
            try:
                # 使用改进的 run_async 来初始化
                run_async(self.browser.init)()
                self._initialized = True
            except Exception as e:
                print(f"Failed to initialize browser: {e}")
                self._initialized = False
                raise

    def navigate_sync(self, url: str) -> str:
        self._ensure_initialized()
        if not self.browser:
            return "Browser not available"
        return run_async(self.browser.navigate)(url)

    def get_page_content_sync(self, clean=True) -> str:
        self._ensure_initialized()
        if not self.browser:
            return "Browser not available"
        return run_async(self.browser.get_page_content)(clean)

    def get_current_url_sync(self) -> str:
        self._ensure_initialized()
        if not self.browser:
            return "Browser not available"
        return self.browser.get_current_url()

    def screenshot_sync(self, full_page: bool = False) -> str:
        self._ensure_initialized()
        if not self.browser:
            return "Browser not available"
        return run_async(self.browser.screenshot)(full_page)

    def close_browser_sync(self) -> str:
        if self._initialized and self.browser:
            try:
                run_async(self.browser.close)()
                self._initialized = False
                return "Browser closed successfully"
            except Exception as e:
                self._initialized = False
                return f"Error closing browser: {e}"
        return "Browser was not initialized"

    def get_tools(self) -> List[StructuredTool]:
        if not self.browser:
            return []
            
        return [
            StructuredTool.from_function(
                func=self.navigate_sync,
                name="navigate_to_url",
                description="Navigate to a specific URL."
            ),
            StructuredTool.from_function(
                func=self.get_page_content_sync,
                name="get_page_content",
                description="Get the text content of the current web page, optionally cleaning it."
            ),
            StructuredTool.from_function(
                func=self.get_current_url_sync,
                name="get_current_url",
                description="Get the current URL of the browser."
            ),
            StructuredTool.from_function(
                func=self.screenshot_sync,
                name="take_screenshot",
                description="Take a screenshot of the current page."
            ),
            StructuredTool.from_function(
                func=self.close_browser_sync,
                name="close_browser",
                description="Close the browser."
            )
        ]

    def __del__(self):
        # 不要在 __del__ 中执行复杂的异步操作，只做简单清理
        if hasattr(self, '_initialized'):
            self._initialized = False
        if hasattr(self, 'browser'):
            self.browser = None 