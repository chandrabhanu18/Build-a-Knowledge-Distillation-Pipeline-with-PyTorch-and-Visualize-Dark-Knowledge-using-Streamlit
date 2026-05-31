from playwright.sync_api import sync_playwright

URL = "http://localhost:8501"

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()
    page.goto(URL, wait_until="networkidle", timeout=120000)
    page.wait_for_timeout(10000)
    page.screenshot(path='tests/live_streamlit_full.png', full_page=True)
    print('Saved screenshot to tests/live_streamlit_full.png')
    browser.close()
