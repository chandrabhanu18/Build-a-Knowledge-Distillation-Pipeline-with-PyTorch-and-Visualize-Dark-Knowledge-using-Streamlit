from playwright.sync_api import sync_playwright

URL = "http://localhost:8501"

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()
    page.goto(URL, wait_until="networkidle", timeout=120000)
    page.wait_for_timeout(5000)
    print(f'Page URL: {page.url}')
    for i, f in enumerate(page.frames):
        try:
            content = f.content()
        except Exception as e:
            content = f'<error: {e}>'
        print('--- FRAME', i, 'name=', f.name, 'url=', f.url)
        snippet = content[:1000].replace('\n', ' ')
        print(snippet)
    browser.close()
