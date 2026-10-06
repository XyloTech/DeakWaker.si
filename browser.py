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
    HEADLESS,
    OBSERVATION_TEXT_LIMIT,
    SLOW_MO_MS,
    USER_DATA_DIR,
)

ACTION_TIMEOUT_MS = 3000
NAVIGATE_TIMEOUT_MS = 15000
SCROLL_STEP_PX = 600

NO_CHANNEL_TOKENS = frozenset({"", "none", "off", "bundled"})
DIAGNOSTICS_LIMIT = 20

_COLLECT_ELEMENTS_JS = """
() => {
  const selectorFor = (el) => {
    if (el.id) return '#' + CSS.escape(el.id);
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
    const raw = ((el.innerText || el.value || el.placeholder || '') + '').trim();
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
    ):
        self.headless = HEADLESS if headless is None else headless
        self.slow_mo = SLOW_MO_MS if slow_mo is None else slow_mo
        self.channel = BROWSER_CHANNEL if channel is None else channel
        self._playwright: Playwright | None = None
        self._browser: BrowserContext | None = None
        self._page: Page | None = None
        self._pages: list[Page] = []
        self._elements: list[dict] = []
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
        channel = _resolve_channel(self.channel)
        try:
            return await self._launch(channel)
        except PlaywrightError as exc:
            if channel is None:
                raise BrowserActionError(f"Failed to start browser: {exc}") from exc
            print(
                f"  ⚠ Google Chrome channel {channel!r} failed "
                f"({exc}) — falling back to bundled Chromium."
            )
            try:
                return await self._launch(None)
            except PlaywrightError as fallback_exc:
                raise BrowserActionError(
                    f"Failed to start browser: {fallback_exc}"
                ) from fallback_exc

    async def _launch(self, channel: str | None) -> BrowserContext:
        return await self._playwright.chromium.launch_persistent_context(
            USER_DATA_DIR, headless=self.headless, slow_mo=self.slow_mo, channel=channel
        )

    async def close(self) -> None:
        browser, playwright = self._browser, self._playwright
        self._browser = None
        self._playwright = None
        self._page = None
        self._pages = []
        self._elements = []
        self._diagnostics.clear()
        try:
            if browser is not None:
                try:
                    await browser.close()
                except PlaywrightError:
                    pass
        finally:
            if playwright is not None:
                await playwright.stop()

    async def navigate_to(self, url: str) -> None:
        page = self._require_page()
        try:
            response = await page.goto(url, timeout=NAVIGATE_TIMEOUT_MS)
        except PlaywrightError as exc:
            raise BrowserActionError(f"Failed to navigate to {url!r}: {exc}") from exc
        self._http_status = response.status if response else None

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
        except PlaywrightError as exc:
            raise BrowserActionError(f"Failed to read page info: {exc}") from exc
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
        }

    async def get_interactive_elements(self) -> list[dict]:
        page = self._require_page()
        try:
            collected = await page.evaluate(_COLLECT_ELEMENTS_JS)
        except PlaywrightError as exc:
            raise BrowserActionError(
                f"Failed to observe interactive elements: {exc}"
            ) from exc
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

    def _require_page(self) -> Page:
        if self._page is None:
            raise BrowserActionError("Browser is not started; call start() first")
        return self._page

    def _resolve(self, index: int) -> dict:
        if index < 0 or index >= len(self._elements):
            raise BrowserActionError(f"Unknown element index: {index}")
        return self._elements[index]
