from playwright.sync_api import sync_playwright
import sys
import os

# Allow overriding the app URL via `APP_URL` environment variable (useful for running inside containers)
URL = os.environ.get("APP_URL", "http://localhost:8501")

selectors = [
    '[data-testid="temp-slider"]',
    '[data-testid="soft-target-inspector"]',
    '[data-testid="soft-target-chart"]',
    '[data-testid="dark-knowledge-viewer"]',
    '[data-testid="compression-dashboard"]',
    '[data-testid="distillation-curve"]',
]


def run_check():
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(URL, wait_until="networkidle", timeout=120000)
            # wait for Streamlit to finish rendering (prerenderReady becomes true)
            try:
                page.wait_for_function("window.prerenderReady === true", timeout=60000)
            except Exception:
                pass

            # helper: search main page then all frames for a selector
            def wait_for_selector_anywhere(page_or_browser, selector, timeout=20000):
                try:
                    page.wait_for_selector(selector, timeout=timeout)
                    return page
                except Exception:
                    pass
                for f in page.frames:
                    try:
                        f.wait_for_selector(selector, timeout=timeout)
                        return f
                    except Exception:
                        continue
                raise Exception(f"Selector not found anywhere: {selector}")

            # ensure all expected selectors are present somewhere on the page or in frames
            for sel in selectors:
                wait_for_selector_anywhere(page, sel, timeout=20000)

            # find the frame/page that holds the slider so we can interact with it
            slider_host = None
            try:
                slider_host = wait_for_selector_anywhere(page, '#temp-slider', timeout=5000)
            except Exception:
                slider_host = page

            before = None
            try:
                host_for_chart = wait_for_selector_anywhere(page, '[data-testid="soft-target-chart"]', timeout=2000)
                before = host_for_chart.inner_html('[data-testid="soft-target-chart"]')
            except Exception:
                before = ''

            # change temperature slider inside the found host
            try:
                slider_host.eval_on_selector('#temp-slider', "el => { el.value = 8; el.dispatchEvent(new Event('input', {bubbles:true})); }")
            except Exception:
                # fallback: try clicking a range input if eval_on_selector fails
                try:
                    slider_host.click('#temp-slider')
                except Exception:
                    pass
            page.wait_for_timeout(1500)
            # re-read chart after change
            try:
                host_for_chart = wait_for_selector_anywhere(page, '[data-testid="soft-target-chart"]', timeout=2000)
                after = host_for_chart.inner_html('[data-testid="soft-target-chart"]')
            except Exception:
                after = ''
            if before.strip() == after.strip():
                print('FAIL: chart did not change after adjusting temperature slider')
                return 2
            # sanity: take screenshot
            page.screenshot(path='tests/playwright_screenshot.png', full_page=True)
            browser.close()
        print('OK: Playwright checks passed')
        return 0
    except Exception as e:
        print('ERROR:', e)
        return 3


if __name__ == '__main__':
    sys.exit(run_check())
