from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    b = p.chromium.launch(headless=True)
    page = b.new_page()
    page.goto('http://localhost:8501', wait_until='networkidle', timeout=120000)
    page.wait_for_timeout(8000)
    frames = page.frames
    for i, f in enumerate(frames):
        print('FRAME', i, 'name', f.name, 'url', f.url)
        try:
            el = f.query_selector('[data-testid="temp-slider"]')
            print('has temp-slider?', bool(el))
        except Exception as e:
            print('err', e)
    b.close()
