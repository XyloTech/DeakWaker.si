import asyncio
import os
import platform
import shutil
from collections import deque
from pathlib import Path

from playwright.async_api import (
    BrowserContext,
    Error as PlaywrightError,
    Page,
    Playwright,
    async_playwright,
)

from config import (
    BROWSER_CHANNEL,
    BROWSER_CHANNEL_ALTERNATIVES,
    BROWSER_ALLOW_FALLBACK,
    BROWSER_CDP_URL,
    BROWSER_EXECUTABLE,
    HEADLESS,
    OBSERVATION_TEXT_LIMIT,
    SLOW_MO_MS,
    USER_DATA_DIR,
    DOWNLOAD_DIR,
)

ACTION_TIMEOUT_MS = 3000
NAVIGATE_TIMEOUT_MS = 15000
OBSERVE_RETRIES = 3
OBSERVE_RETRY_DELAY_S = 0.2
SCROLL_STEP_PX = 600
NO_CHANNEL_TOKENS = frozenset({"", "none", "off", "bundled"})
SYSTEM_CHROMIUM_CHANNEL = "system-chromium"
BRAVE_CHANNEL = "brave"

DIAGNOSTICS_LIMIT = 20


def detect_browsers() -> list[dict[str, str | bool]]:
    """Return installed browser candidates without changing the selected browser."""
    candidates: list[dict[str, str | bool]] = []
    known = {
        "brave": [
            "/usr/bin/brave-browser",
            "/usr/bin/brave-browser-stable",
            "/opt/brave.com/brave/brave",
            "/opt/brave.com/brave/brave-browser",
            r"C:\Program Files\BraveSoftware\Brave-Browser\Application\brave.exe",
            r"C:\Program Files (x86)\BraveSoftware\Brave-Browser\Application\brave.exe",
        ],
        "chrome": [
            "/usr/bin/google-chrome",
            "/usr/bin/google-chrome-stable",
            "/opt/google/chrome/chrome",
            r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        ],
        "chromium": ["/usr/bin/chromium", "/usr/bin/chromium-browser"],
        "msedge": [
            "/usr/bin/microsoft-edge",
            r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        ],
        "firefox": ["/usr/bin/firefox", r"C:\Program Files\Mozilla Firefox\firefox.exe"],
    }
    local_appdata = os.getenv("LOCALAPPDATA")
    if local_appdata:
        known["brave"].append(str(Path(local_appdata) / "BraveSoftware/Brave-Browser/Application/brave.exe"))
    path_names = {"brave": "brave-browser", "chrome": "google-chrome", "chromium": "chromium", "msedge": "microsoft-edge", "firefox": "firefox"}
    for name, paths in known.items():
        path = next((item for item in paths if Path(item).exists()), None)
        executable = path or shutil.which(path_names[name])
        if executable:
            candidates.append(
                {
                    "name": name,
                    "executable": executable,
                    "playwright_channel": name if name in {"chrome", "msedge", "firefox"} else "",
                    "usable_here": not (platform.system() == "Linux" and executable.lower().endswith(".exe")),
                }
            )
    if "microsoft" in platform.release().lower() or os.getenv("WSL_DISTRO_NAME"):
        for name, executable in (
            ("brave-windows-host", r"/mnt/c/Program Files/BraveSoftware/Brave-Browser/Application/brave.exe"),
            ("chrome-windows-host", r"/mnt/c/Program Files/Google/Chrome/Application/chrome.exe"),
            ("edge-windows-host", r"/mnt/c/Program Files (x86)/Microsoft/Edge/Application/msedge.exe"),
        ):
            if Path(executable).exists() and not any(item["executable"] == executable for item in candidates):
                candidates.append({"name": name, "executable": executable, "playwright_channel": "", "usable_here": False})
    return candidates


def _find_installed_executable(name: str) -> str | None:
    for browser in detect_browsers():
        if browser["name"] == name and browser.get("usable_here", True):
            return str(browser["executable"])
    return None


_COLLECT_ELEMENTS_JS = """
() => {
  const selectorFor = (el) => {
    if (el.id) {
      const escapedId = '#' + CSS.escape(el.id);
      if (document.querySelectorAll(escapedId).length === 1) return escapedId;
    }
    const parts = [];
    let node = el;
    while (node && node.nodeType === 1 && parts.length < 5) {
      if (node.id) {
        parts.unshift('#' + CSS.escape(node.id));
        break;
      }
      let part = node.tagName.toLowerCase();
      const parent = node.parentElement;
      if (parent) {
        const sameTag = Array.from(parent.children).filter(
          (child) => child.tagName === node.tagName
        );
        if (sameTag.length > 1) {
          part += ':nth-of-type(' + (sameTag.indexOf(node) + 1) + ')';
        }
      }
      parts.unshift(part);
      node = parent;
    }
    return parts.join(' > ');
  };

  const roleFor = (el) => {
    const explicit = el.getAttribute('role');
    if (explicit) return explicit;
    const tag = el.tagName.toLowerCase();
    if (tag === 'a') return 'link';
    if (tag === 'button') return 'button';
    if (tag === 'select') return 'combobox';
    if (tag === 'textarea') return 'textbox';
    if (tag === 'input') {
      const type = (el.type || 'text').toLowerCase();
      if (['submit', 'button', 'reset', 'image'].includes(type)) return 'button';
      if (type === 'password') return 'password';
      if (type === 'checkbox') return 'checkbox';
      if (type === 'radio') return 'radio';
      return 'textbox';
    }
    return tag;
  };

  const textFor = (el) => {
    const raw = ((el.innerText || el.value || el.placeholder ||
      el.getAttribute('aria-label') || el.getAttribute('title') || '') + '').trim();
    return raw.length > 80 ? raw.slice(0, 80) : raw;
  };

  const nodes = document.querySelectorAll(
    'a, button, input, select, textarea, [role=button], [role=link]'
  );
    const collected = [];
    nodes.forEach((el) => {
      if (el.disabled) return;
      if (el.type === 'hidden') return;
      const style = window.getComputedStyle(el);
      if (style.display === 'none' || style.visibility === 'hidden') return;
      const rect = el.getBoundingClientRect();
      if (rect.width <= 0 || rect.height <= 0) return;
      collected.push({
        role: roleFor(el),
        text: textFor(el),
        selector: selectorFor(el),
        submit:
          (el.type || '').toLowerCase() === 'submit' && el.form != null,
      });
    });
    return collected;
}
"""


class BrowserActionError(Exception):
    pass


def _resolve_channel(value: str) -> str | None:
    if value.strip().lower() in NO_CHANNEL_TOKENS:
        return None
    return value


class BrowserWrapper:
    def __init__(
        self,
        *,
        headless: bool | None = None,
        slow_mo: int | None = None,
        channel: str | None = None,
        profile_dir: str | None = None,
        executable_path: str | None = None,
    ):
        self.headless = HEADLESS if headless is None else headless
        self.slow_mo = SLOW_MO_MS if slow_mo is None else slow_mo
        self.channel = BROWSER_CHANNEL if channel is None else channel
        self.profile_dir = str(Path(USER_DATA_DIR if profile_dir is None else profile_dir).expanduser())
        self.executable_path = executable_path
        self.launched_browser = "not started"
        self.last_navigation_url = "about:blank"
        self._playwright: Playwright | None = None
        self._browser: BrowserContext | None = None
        self._cdp_browser = None
        self._page: Page | None = None
        self._pages: list[Page] = []
        self._elements: list[dict] = []
        self._downloads: list[str] = []
        self._http_status: int | None = None
        self._diagnostics: deque[str] = deque(maxlen=DIAGNOSTICS_LIMIT)

    async def start(self) -> None:
        if self._page is not None and not self._page.is_closed():
            return
        if self._playwright is not None:
            await self.close()
        try:
            self._playwright = await async_playwright().start()
        except PlaywrightError as exc:
            raise BrowserActionError(f"Failed to start browser: {exc}") from exc
        Path(DOWNLOAD_DIR).mkdir(parents=True, exist_ok=True)
        self._browser = await self._open_context()
        self._browser.on("page", self._register_page)
        if self._browser.pages:
            self._page = self._browser.pages[0]
        else:
            self._page = await self._browser.new_page()
        self._register_page(self._page)

    def _register_page(self, page: Page) -> None:
        if page in self._pages:
            return
        self._pages.append(page)
        self._attach_page_listeners(page)
        page.on(
            "close",
            lambda: self._pages.remove(page) if page in self._pages else None,
        )

    def _attach_page_listeners(self, page: Page) -> None:
        def record(entry: str) -> None:
            self._diagnostics.append(entry)

        async def on_console(message) -> None:
            if message.type in ("error", "warning"):
                record(f"console.{message.type}: {message.text}")

        async def on_page_error(error) -> None:
            record(f"pageerror: {error}")

        async def on_request_failed(request) -> None:
            failure = request.failure or "unknown"
            record(f"requestfailed: {request.url} — {failure}")

        async def on_response(response) -> None:
            if response.status >= 400:
                record(f"http {response.status}: {response.url}")

        async def on_navigation_response(response) -> None:
            request = response.request
            if request.is_navigation_request() and request.frame == page.main_frame:
                self._http_status = response.status

        async def on_download(download) -> None:
            path = str(Path(DOWNLOAD_DIR) / Path(download.suggested_filename).name)
            await download.save_as(path)
            self._downloads.append(path)
            if len(self._downloads) > 10:
                self._downloads = self._downloads[-10:]

        page.on("download", on_download)

        page.on("console", on_console)
        page.on("pageerror", on_page_error)
        page.on("requestfailed", on_request_failed)
        page.on("response", on_response)
        page.on("response", on_navigation_response)

    def get_diagnostics(self) -> list[str]:
        return list(self._diagnostics)

    def clear_diagnostics(self) -> None:
        self._diagnostics.clear()

    async def _open_context(self) -> BrowserContext:
        if BROWSER_CDP_URL:
            try:
                self._cdp_browser = await self._playwright.chromium.connect_over_cdp(
                    BROWSER_CDP_URL
                )
                if not self._cdp_browser.contexts:
                    raise BrowserActionError(
                        f"No browser context available at BROWSER_CDP_URL={BROWSER_CDP_URL!r}"
                    )
                self.launched_browser = f"Brave (manual CDP: {BROWSER_CDP_URL})"
                return self._cdp_browser.contexts[0]
            except (PlaywrightError, BrowserActionError) as exc:
                raise BrowserActionError(
                    f"Could not connect to manual browser at {BROWSER_CDP_URL!r}: {exc}"
                ) from exc
        channel = _resolve_channel(self.channel)
        if channel is None:
            launch_order = [None]
        elif self.executable_path or BROWSER_EXECUTABLE:
            launch_order = [channel]
        elif not BROWSER_ALLOW_FALLBACK:
            launch_order = [channel]
        else:
            launch_order = [channel] + [
                alternative.strip().lower()
                for alternative in BROWSER_CHANNEL_ALTERNATIVES
                if alternative.strip().lower() != channel.lower()
            ]
        last_exc: Exception | None = None
        for try_channel in launch_order:
            try:
                return await self._launch(try_channel)
            except PlaywrightError as exc:
                last_exc = exc
                continue
        if channel is None or not BROWSER_ALLOW_FALLBACK:
            installed = ", ".join(str(item["name"]) for item in detect_browsers()) or "none detected"
            detail = str(last_exc)
            if "existing browser session" in detail.lower() or "already in use" in detail.lower():
                detail = (
                    f"Profile {self.profile_dir!r} is already in use by another browser process. "
                    "Close that process or choose another USER_DATA_DIR; the agent will not "
                    "silently share or replace the profile."
                )
            raise BrowserActionError(
                f"Failed to start browser {self.channel!r}: {detail}. "
                f"Profile: {self.profile_dir}. Installed here: {installed}. "
                "Choose BROWSER_CHANNEL=none for bundled Chromium or install the selected browser."
            ) from last_exc
        print(
            f"  ⚠ Channel {channel!r} and alternatives failed "
            f"({last_exc}) — falling back to bundled Chromium."
        )
        try:
            return await self._launch(None)
        except PlaywrightError as fallback_exc:
            raise BrowserActionError(
                f"Failed to start browser: {fallback_exc}"
            ) from fallback_exc

    async def _launch(self, channel: str | None) -> BrowserContext:
        options = {
            "headless": self.headless,
            "slow_mo": self.slow_mo,
            "args": ["--disable-http2"],
        }
        if channel is not None and channel != SYSTEM_CHROMIUM_CHANNEL:
            options["channel"] = channel
        executable_path = self.executable_path or BROWSER_EXECUTABLE
        if channel == SYSTEM_CHROMIUM_CHANNEL and not executable_path:
            executable_path = shutil.which("chromium") or "/usr/bin/chromium"
        if channel == BRAVE_CHANNEL and not executable_path:
            executable_path = _find_installed_executable(BRAVE_CHANNEL)
            if not executable_path:
                if os.getenv("WSL_DISTRO_NAME") and any(
                    item["name"] == "brave-windows-host" for item in detect_browsers()
                ):
                    raise PlaywrightError(
                        "Windows Brave was detected, but Linux Playwright cannot control "
                        "that Windows process from this WSL environment; run 'py chat.py' "
                        "in Windows or install Brave inside Kali"
                    )
                raise PlaywrightError(
                    "Brave browser was requested but no usable Brave executable was detected"
                )
        if executable_path:
            options["executable_path"] = executable_path
            options.pop("channel", None)
        self.launched_browser = executable_path or channel or "bundled Chromium"
        return await self._playwright.chromium.launch_persistent_context(
            self.profile_dir, **options
        )

    def describe(self) -> str:
        return f"browser={self.launched_browser}; profile={Path(self.profile_dir).resolve()}"

    async def close(self) -> None:
        browser, playwright, cdp_browser = self._browser, self._playwright, self._cdp_browser
        self._browser = None
        self._cdp_browser = None
        self._playwright = None
        self._page = None
        self._pages = []
        self._elements = []
        self._diagnostics.clear()
        self._downloads.clear()
        try:
            if browser is not None and cdp_browser is None:
                try:
                    await browser.close()
                except PlaywrightError:
                    pass
            if cdp_browser is not None:
                try:
                    await cdp_browser.close()
                except PlaywrightError:
                    pass
        finally:
            if playwright is not None:
                await playwright.stop()

    async def navigate_to(self, url: str) -> dict:
        page = self._require_page()
        self.last_navigation_url = url
        if url == "about:blank":
            try:
                await page.goto(url, timeout=NAVIGATE_TIMEOUT_MS, wait_until="commit")
                await page.set_content("<!doctype html><html><head></head><body></body></html>")
                if page.url != "about:blank":
                    raise BrowserActionError(
                        f"Blank navigation did not settle; current_url={page.url}"
                    )
                self._http_status = None
                self._elements = []
                return {"http_status": None, "response": None, "url": page.url}
            except PlaywrightError as exc:
                raise BrowserActionError(
                    f"Failed to navigate to {url!r}: {exc}; current_url={page.url}"
                ) from exc
        candidates = [url]
        if "www.goindigo.com" in url:
            candidates.append(url.replace("www.goindigo.com", "www.goindigo.in"))
        last_exc: PlaywrightError | None = None
        response = None
        succeeded = False
        for candidate in candidates:
            for attempt in range(2):
                try:
                    response = await page.goto(
                        candidate,
                        timeout=NAVIGATE_TIMEOUT_MS,
                        wait_until="domcontentloaded" if attempt == 0 else "commit",
                    )
                    break
                except PlaywrightError as exc:
                    last_exc = exc
                    if attempt == 0:
                        await asyncio.sleep(0.5)
                        continue
                    break
            else:
                continue
            current_url = getattr(page, "url", "") or ""
            if response is not None or (candidate == "about:blank" and current_url == "about:blank"):
                url = candidate
                succeeded = True
                self._elements = []
                break
        if not succeeded:
            current_url = getattr(page, "url", "about:blank") or "about:blank"
            if current_url != "about:blank":
                return {
                    "http_status": self._http_status,
                    "response": None,
                    "url": current_url,
                    "recovered": True,
                }
            raise BrowserActionError(
                f"Failed to navigate to {url!r}: {last_exc}; current_url={current_url}"
            ) from last_exc
        self._http_status = response.status if response else None
        # Return status info for error handling
        return {"http_status": self._http_status, "response": response}

    async def handle_http_error(self, http_status: int | None) -> str | None:
        """Detect and return error type if the HTTP status indicates a known issue."""
        if http_status is None:
            return None
        if http_status == 404:
            return "404_not_found"
        if http_status == 403:
            return "403_forbidden"
        if http_status == 401:
            return "401_unauthorized"
        if http_status == 429:
            return "429_rate_limited"
        if http_status >= 500:
            return "5xx_server_error"
        return None

    async def dismiss_cookie_banner(self) -> bool:
        """Try to dismiss a cookie consent banner if present."""
        try:
            element_handlers = [
                'button:has-text("Accept")',
                'button:has-text("I agree")',
                'button:has-text("Accept all")',
                '[role="button"]:has-text("Accept")',
                '.cookie-accept',
                '#accept-cookies',
                'button[data-cookie-accept]',
            ]
            for selector in element_handlers:
                elements = await self._page.query_selector_all(selector)
                for el in elements:
                    if await el.is_visible():
                        await el.click()
                        await asyncio.sleep(0.3)
                        return True
        except PlaywrightError:
            pass
        return False

    async def dismiss_popups(self) -> bool:
        """Try to dismiss common pop-ups/modals."""
        try:
            selectors = [
                'button:has-text("Close")',
                'button:has-text("Dismiss")',
                'button:has-text("Skip")',
                '[role="dialog"] button',
                '.modal-close',
                '.popup-close',
            ]
            for selector in selectors:
                elements = await self._page.query_selector_all(selector)
                for el in elements:
                    if await el.is_visible():
                        await el.click()
                        await asyncio.sleep(0.3)
                        return True
        except PlaywrightError:
            pass
        return False

    async def new_tab(self, url: str | None = None) -> None:
        if self._browser is None:
            raise BrowserActionError("Browser is not started; call start() first")
        page = await self._browser.new_page()
        self._register_page(page)
        self._page = page
        if url is None:
            return
        try:
            await page.goto(url, timeout=NAVIGATE_TIMEOUT_MS)
        except PlaywrightError as exc:
            raise BrowserActionError(f"Failed to navigate to {url!r}: {exc}") from exc

    async def switch_tab(self, index: int) -> None:
        if index < 0 or index >= len(self._pages):
            raise BrowserActionError(
                f"no tab {index}; {len(self._pages)} tabs open"
            )
        self._page = self._pages[index]

    @property
    def tab_count(self) -> int:
        return len(self._pages)

    async def get_page_info(self) -> dict:
        page = self._require_page()
        last_error: PlaywrightError | None = None
        for attempt in range(OBSERVE_RETRIES):
            try:
                info = await page.evaluate(
                    """() => ({
                        url: window.location.href,
                        title: document.title,
                        text: document.body ? document.body.innerText : '',
                        has_password:
                            document.querySelector('input[type="password"]') !== null,
                    })"""
                )
                break
            except PlaywrightError as exc:
                last_error = exc
                if attempt + 1 == OBSERVE_RETRIES:
                    raise BrowserActionError(f"Failed to read page info: {exc}") from exc
                await asyncio.sleep(OBSERVE_RETRY_DELAY_S)
        else:
            raise BrowserActionError(f"Failed to read page info: {last_error}")
        return {
            "url": str(info["url"]),
            "title": str(info["title"]),
            "text": str(info["text"] or "")[:OBSERVATION_TEXT_LIMIT],
            "has_password": bool(info["has_password"]),
            "http_status": self._http_status,
            "tabs": len(self._pages),
            "active_tab": (
                self._pages.index(self._page) if self._page in self._pages else 0
            ),
            "recent_downloads": list(self._downloads[-3:]),
        }

    async def get_interactive_elements(self) -> list[dict]:
        page = self._require_page()
        last_error: PlaywrightError | None = None
        for attempt in range(OBSERVE_RETRIES):
            try:
                collected = await page.evaluate(_COLLECT_ELEMENTS_JS)
                break
            except PlaywrightError as exc:
                last_error = exc
                if attempt + 1 == OBSERVE_RETRIES:
                    raise BrowserActionError(
                        f"Failed to observe interactive elements: {exc}"
                    ) from exc
                await asyncio.sleep(OBSERVE_RETRY_DELAY_S)
        else:
            raise BrowserActionError(
                f"Failed to observe interactive elements: {last_error}"
            )
        self._elements = [
            {
                "index": index,
                "role": item["role"],
                "text": item["text"],
                "selector": item["selector"],
                "submit": bool(item.get("submit", False)),
            }
            for index, item in enumerate(collected)
        ]
        return list(self._elements)

    async def click(self, index: int) -> None:
        if self.last_navigation_url == "about:blank":
            raise BrowserActionError("Cannot click: the active page is about:blank")
        element = self._resolve(index)
        page = self._require_page()
        try:
            await page.click(element["selector"], timeout=ACTION_TIMEOUT_MS)
        except PlaywrightError as exc:
            raise BrowserActionError(
                f"click({index}) failed for {element['selector']!r}: {exc}"
            ) from exc

    async def type_text(self, index: int, text: str) -> None:
        element = self._resolve(index)
        page = self._require_page()
        try:
            await page.fill(element["selector"], text, timeout=ACTION_TIMEOUT_MS)
        except PlaywrightError as exc:
            raise BrowserActionError(
                f"type_text({index}) failed for {element['selector']!r}: {exc}"
            ) from exc

    async def select_option(self, index: int, value: str) -> None:
        element = self._resolve(index)
        selector = element["selector"]
        page = self._require_page()
        try:
            await page.select_option(selector, value=value, timeout=ACTION_TIMEOUT_MS)
            return
        except PlaywrightError:
            pass
        try:
            await page.select_option(selector, label=value, timeout=ACTION_TIMEOUT_MS)
            return
        except PlaywrightError:
            pass
        try:
            opts = await page.evaluate(
                f"sel => Array.from(document.querySelector(sel).options).map(o => o.value + '|' + o.text)",
                selector,
            )
            raise BrowserActionError(
                f"select_option({index}) failed: no option matching {value!r}; available: {opts}"
            )
        except PlaywrightError:
            raise BrowserActionError(
                f"select_option({index}) failed: no option matching {value!r}"
            )

    async def upload_file(self, index: int, path: str) -> None:
        if not Path(path).is_file():
            raise BrowserActionError(f"upload_file: no such file: {path}")
        element = self._resolve(index)
        selector = element["selector"]
        page = self._require_page()
        try:
            await page.set_input_files(selector, str(path), timeout=ACTION_TIMEOUT_MS)
        except PlaywrightError as exc:
            raise BrowserActionError(
                f"upload_file({index}) failed for {selector!r}: {exc}"
            ) from exc

    async def get_value(self, index: int) -> str:
        element = self._resolve(index)
        page = self._require_page()
        try:
            return await page.input_value(element["selector"], timeout=ACTION_TIMEOUT_MS)
        except PlaywrightError as exc:
            raise BrowserActionError(
                f"get_value({index}) failed for {element['selector']!r}: {exc}"
            ) from exc

    async def scroll(self, direction: str = "down") -> None:
        page = self._require_page()
        step = {"down": SCROLL_STEP_PX, "up": -SCROLL_STEP_PX}.get(direction)
        if step is None:
            raise BrowserActionError(f"Unknown scroll direction: {direction!r}")
        try:
            await page.evaluate("step => window.scrollBy(0, step)", step)
        except PlaywrightError as exc:
            raise BrowserActionError(f"Failed to scroll: {exc}") from exc

    async def get_scroll_position(self) -> dict:
        page = self._require_page()
        try:
            position = await page.evaluate(
                """() => ({
                    y: Math.round(window.scrollY),
                    max: Math.max(
                        0,
                        document.documentElement.scrollHeight - window.innerHeight
                    ),
                })"""
            )
        except PlaywrightError as exc:
            raise BrowserActionError(f"Failed to read scroll position: {exc}") from exc
        return {"y": int(position["y"]), "max": int(position["max"])}

    async def take_screenshot(self, path: str) -> None:
        page = self._require_page()
        try:
            await page.screenshot(path=path)
        except PlaywrightError as exc:
            raise BrowserActionError(f"Failed to save screenshot to {path!r}: {exc}") from exc

    async def get_page_text_safe(self) -> str:
        """Get page text, trying to reveal hidden content by scrolling first."""
        await self.scroll("down")
        await asyncio.sleep(0.3)
        await self.scroll("up")
        await asyncio.sleep(0.3)
        return await self.get_page_info()

    async def wait_for_load(self, timeout: float | None = None) -> bool:
        """Wait for the page to finish loading."""
        if timeout is None:
            timeout = NAVIGATE_TIMEOUT_MS / 1000
        try:
            await self._page.wait_for_load_state("networkidle", timeout=timeout * 1000)
            return True
        except PlaywrightError:
            try:
                await self._page.wait_for_load_state("domcontentloaded", timeout=timeout * 1000)
                return True
            except PlaywrightError:
                return False

    async def smart_click(self, index: int, retries: int = 2) -> str:
        """Click an element with retry and scroll-into-view if needed."""
        if self.last_navigation_url == "about:blank":
            return "ERROR: Cannot click: the active page is about:blank"
        element = self._resolve(index)
        page = self._require_page()
        for attempt in range(retries + 1):
            try:
                locator = page.locator(element["selector"])
                if await locator.count() != 1:
                    text = str(element.get("text", "")).strip()
                    role = str(element.get("role", "")).strip()
                    if text and role in {"button", "link", "tab", "checkbox", "radio"}:
                        semantic = page.get_by_role(role, name=text, exact=False)
                        if await semantic.count():
                            locator = semantic.first
                    elif text:
                        semantic = page.get_by_text(text, exact=False)
                        if await semantic.count():
                            locator = semantic.first
                    else:
                        locator = locator.first
                await locator.scroll_into_view_if_needed(timeout=ACTION_TIMEOUT_MS)
                await asyncio.sleep(0.2)
                await locator.click(timeout=ACTION_TIMEOUT_MS)
                return "OK"
            except PlaywrightError as exc:
                if attempt < retries:
                    await self.scroll("down")
                    await asyncio.sleep(0.3)
                    continue
                return f"ERROR: click failed after {retries} retries: {exc}"
        return "ERROR: unknown"

    def _require_page(self) -> Page:
        if self._page is None:
            raise BrowserActionError("Browser is not started; call start() first")
        return self._page

    def _resolve(self, index: int) -> dict:
        if index < 0 or index >= len(self._elements):
            raise BrowserActionError(f"Unknown element index: {index}")
        return self._elements[index]
