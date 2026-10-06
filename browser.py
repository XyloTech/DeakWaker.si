from playwright.async_api import (
    Browser,
    Error as PlaywrightError,
    Page,
    Playwright,
    async_playwright,
)

from config import HEADLESS, OBSERVATION_TEXT_LIMIT

ACTION_TIMEOUT_MS = 3000
NAVIGATE_TIMEOUT_MS = 15000
SCROLL_STEP_PX = 600

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
    collected.push({ role: roleFor(el), text: textFor(el), selector: selectorFor(el) });
  });
  return collected;
}
"""


class BrowserActionError(Exception):
    pass


class BrowserWrapper:
    def __init__(self, *, headless: bool = HEADLESS):
        self.headless = headless
        self._playwright: Playwright | None = None
        self._browser: Browser | None = None
        self._page: Page | None = None
        self._elements: list[dict] = []

    async def start(self) -> None:
        if self._page is not None:
            return
        try:
            self._playwright = await async_playwright().start()
            self._browser = await self._playwright.chromium.launch(headless=self.headless)
            self._page = await self._browser.new_page()
        except PlaywrightError as exc:
            raise BrowserActionError(f"Failed to start browser: {exc}") from exc

    async def close(self) -> None:
        browser, playwright = self._browser, self._playwright
        self._browser = None
        self._playwright = None
        self._page = None
        self._elements = []
        try:
            if browser is not None:
                await browser.close()
        finally:
            if playwright is not None:
                await playwright.stop()

    async def navigate_to(self, url: str) -> None:
        page = self._require_page()
        try:
            await page.goto(url, timeout=NAVIGATE_TIMEOUT_MS)
        except PlaywrightError as exc:
            raise BrowserActionError(f"Failed to navigate to {url!r}: {exc}") from exc

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
