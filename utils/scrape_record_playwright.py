import time
import re
from playwright.sync_api import sync_playwright
from playwright_stealth import stealth_sync

DEBUG = True  # Set to True to visually watch the browser scrape

def get_record_text_from_communication_page(comm_url: str, ec_id: str) -> str:
    with sync_playwright() as p:
        print(f"🌐 Loading: {comm_url}")
        browser = p.chromium.launch(headless=not DEBUG)
        page = browser.new_page()
        stealth_sync(page)

        try:
            page.goto(comm_url, timeout=60000)
            print("🕵️ Stealth mode applied. Waiting for page load...")
            time.sleep(2)

            # Click "More Details" if available
            tab_button = page.locator("button:has-text('More Details')")
            if tab_button.count() > 0:
                print("🖱 Clicking 'More Details' tab...")
                tab_button.first.click()
                time.sleep(2)

            # Search for Congressional Record link
            print("🔍 Searching all links on the page...")
            anchors = page.locator("a")
            count = anchors.count()

            for i in range(count):
                href = anchors.nth(i).get_attribute("href")
                if href and "/congressional-record/" in href and "/article/" in href:
                    full_url = "https://www.congress.gov" + href
                    print(f"➡️ Found Congressional Record link: {full_url}")
                    page.goto(full_url, timeout=60000)

                    try:
                        print("🔍 Waiting for <pre> block...")
                        page.wait_for_selector("pre", timeout=10000)
                        pre = page.locator("pre")
                        full_text = pre.text_content().strip() if pre else ""

                        # Extract the EC ID paragraph only
                        match = re.search(rf"(EC[-–]{ec_id}\..*?)(?=\n\n|\nEC-|\Z)", full_text, re.DOTALL)
                        text = match.group(1).strip() if match else ""

                        if text:
                            print(f"📄 Final extracted text for EC-{ec_id}:\n{text[:1000]}\n")
                        else:
                            print(f"❌ EC-{ec_id} not found in text.\n")

                        browser.close()
                        return text[:3000] if text else f"❌ EC-{ec_id} not found in record."
                    except Exception as e:
                        print(f"⚠️ Error extracting EC-{ec_id}: {e}")
                        browser.close()
                        return f"❌ EC-{ec_id} not found or page failed."
        except Exception as e:
            print(f"❌ Exception during scrape: {e}")
        finally:
            browser.close()

        return f"❌ EC-{ec_id} not found (fallback)."
