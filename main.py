import json
import os
import pickle
import time
import traceback
from collections import Counter
from turtle import window_width
from typing import Iterable, TypedDict

import cv2
import numpy as np
import pyautogui
import pygetwindow
import requests
import win32gui  # type: ignore
from bs4 import BeautifulSoup
from pyscreeze import pixel, screenshot
from tqdm import tqdm

hasher = cv2.img_hash.PHash_create() #type: ignore

class Support(TypedDict):
    url_name: str
    support_id: int
    char_id: int
    char_name: str
    name_jp: str
    name_ko: str
    name_tw: str
    rarity: int
    type: str
    obtained: str
    release: str
    release_ko: str
    release_zh_tw: str
    release_en: str 
    effects: list[list[int]]
    hints: dict
    event_skills: list[dict]
    hash: str


def fetch_gametora_build_id() -> str:
    url = "https://gametora.com/"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Find the script tag with id "__NEXT_DATA__"
    script_tag = soup.find('script', id='__NEXT_DATA__')
    
    if script_tag:
        script_content = script_tag.get_text()
        if script_content:
            try:
                # Parse the JSON content
                json_data = json.loads(script_content)
                # Extract the buildId
                build_id = json_data.get('buildId')
                return build_id
            except json.JSONDecodeError as e:
                raise RuntimeError(f"Error parsing JSON: {e}")
        else:
            raise RuntimeError("Script tag has no content")
    else:
        raise RuntimeError("Script tag with id '__NEXT_DATA__' not found")


def fetch_gametora_supports(build_id: str) -> list[Support] | None:
    url = f"https://gametora.com/_next/data/{build_id}/umamusume/supports.json"
    response = requests.get(url)
    data = response.json()
    
    try:
        if response.status_code != 200:
            raise RuntimeError(f"Failed to fetch supports. Status code: {response.status_code}")
            
        if not data or 'pageProps' not in data:
            raise RuntimeError("Invalid response format - missing pageProps")
            
        supports = data.get('pageProps', {}).get('supportData')
        if not supports:
            raise RuntimeError("No support data found in response")
            
        return supports
        
    except requests.RequestException as e:
        raise RuntimeError(f"Request error: {e}")
    except json.JSONDecodeError as e:
        raise RuntimeError(f"JSON decode error: {e}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error: {e}")

def fetch_gametora_image(path: str):
    try:
        res = requests.get(f'https://gametora.com/images/umamusume/{path}', timeout=10)
        res.raise_for_status()
        image = np.frombuffer(res.content, dtype=np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Failed to decode image from {path}")
        return image
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Failed to download image from {path}. Reason: {e}")

def compare_hashes(hash1: str, hash2: str):
    return bin(int(hash1, 16) ^ int(hash2, 16)).count('1')

def digest_hash(hash: np.ndarray):
    return ''.join(f'{x:02x}' for x in hash.flatten())

def compute_hash(image: np.ndarray, hasher = hasher):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hash = hasher.compute(gray)
    return digest_hash(hash)

def crop(img: np.ndarray, region: tuple[int, int, int, int]):
    x, y, w, h = region
    return img[y:y+h, x:x+w]

def extract_brown(img_bgr: np.ndarray):
    """
    Extracts brown text tolor from image.
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # +/- 3 from 13 hue, 40-255 saturation, +/- 20 from 120 value
    lower_brown = np.array([10, 40, 100])
    upper_brown = np.array([16, 255, 140])

    mask = cv2.inRange(hsv, lower_brown, upper_brown)

    return mask

def scale_region(region: tuple[int, int, int, int], scale_x: int, scale_y: int):
    x, y, w, h = region
    return (
        round(x * scale_x),
        round(y * scale_y),
        round(w * scale_x),
        round(h * scale_y)
    )

class Agent:
    CARD_WIDTH = 121
    CARD_HEIGHT = 157
    CARD_ROI = (27, 32, 68, 94)
    WINDOW_SIZE = (1920, 1080)

    supports: list[Support]
    window_ensurance: bool

    def __init__(self, window_title: str):
        self.window_title = window_title
        self.hwnd = win32gui.FindWindow(None, window_title)
        if not self.hwnd:
            raise RuntimeError(f"Window {window_title} not found")
        
        self.window = pygetwindow.getWindowsWithTitle(window_title)[0]
        try:
            self.window.activate()
        except pygetwindow.PyGetWindowException as e:
            if 'The operation completed successfully' in str(e):
                pass
            else:
                raise e
        self.window_ensurance = True
        self.ensure_window()
        self.supports = self.load_supports()

    def load_supports(self) -> list[Support]:
        try:
            with open('supports.pkl', 'rb') as f:
                return pickle.load(f)
        except (FileNotFoundError, EOFError):
            supports = self.fetch_supports()
            with open('supports.pkl', 'wb') as f:
                pickle.dump(supports, f)
            return supports

    def fetch_supports(self):
        build_id = fetch_gametora_build_id()
        if build_id is None:
            raise RuntimeError("Failed to fetch build id")
            
        supports = fetch_gametora_supports(build_id)
        if supports is None:
            raise RuntimeError("Failed to fetch supports")

        for support in tqdm(supports, desc="Fetching and hashing cards"):
            image = fetch_gametora_image(f'supports/tex_support_card_{support["support_id"]}.png')
            if image is None:
                raise RuntimeError(f"Failed to fetch image for {support['url_name']}")
            hash = Agent.hash_card(image)
            support['hash'] = hash

        return supports

    @staticmethod
    def hash_card(image: np.ndarray):
        h, w = image.shape[:2]
        cropped = crop(image, scale_region(Agent.CARD_ROI, w / Agent.CARD_WIDTH, h / Agent.CARD_HEIGHT))
        return compute_hash(cropped)
    
    def client_to_screen(self, x_offset: int, y_offset: int):
        return win32gui.ClientToScreen(self.hwnd, (x_offset, y_offset))
        # return self.window.left + x_offset + 8, self.window.top + y_offset + 31

    def click(self, x_offset: int, y_offset: int):
        self.ensure_window()
        x, y = self.client_to_screen(x_offset, y_offset)
        screen_width, screen_height = pyautogui.size()
        if not (0 <= x < screen_width and 0 <= y < screen_height):
            return
        pyautogui.click(x, y)
        time.sleep(0.1)

    def extract_cards(self):
        screenshotg = self.capture()
        roi = crop(screenshotg, (308, 111, 481, 746))
        for x in [0, 179, 360]:
            for y in [0, 392]:
                yield crop(roi, (x, y, Agent.CARD_WIDTH, Agent.CARD_HEIGHT))

        for x in [90, 271]:
            for y in [195, 589]:
                yield crop(roi, (x, y, Agent.CARD_WIDTH, Agent.CARD_HEIGHT))
    
    def extract_single_card(self):
        screenshot = self.capture()
        card =  crop(screenshot, (488, 384, Agent.CARD_WIDTH, Agent.CARD_HEIGHT))
        yield card
        
    def read_rolls(self, cards: Iterable[np.ndarray]):
        for card in cards:
                closest_match = self.supports[0]
                closest_match_distance = float('inf')

                roi = crop(card, Agent.CARD_ROI)
                hash = compute_hash(roi)

                for support in self.supports:
                    distance = compare_hashes(hash, support['hash'])
                    if distance < closest_match_distance:
                        closest_match = support
                        closest_match_distance = distance

                # if closest_match_distance > 1:  # Only update if hash distance is greater than 1
                #     closest_match['hash'] = hash  # Update hash in memory
                #     with open('supports.pkl', 'wb') as f:  # Update pickle file
                #         pickle.dump(self.supports, f)  # Save updated supports

                yield closest_match, closest_match_distance
    
    def set_window_ensurance(self, value: bool):
        self.window_ensurance = value

    def ensure_window(self, timeout = 10, poll_interval = 0.05):
        if not self.window_ensurance:
            return
        width, height = Agent.WINDOW_SIZE
        left, top = 0, 0
        win32gui.MoveWindow(self.hwnd, left, top, width, height, True)
        end_time = time.time() + timeout
        while time.time() < end_time:
            rect = win32gui.GetWindowRect(self.hwnd)
            x, y, w, h = rect[0], rect[1], rect[2] - rect[0], rect[3] - rect[1]
            if (x, y) == (left, top) and (w, h) == (width, height):
                return 
            time.sleep(poll_interval)
        raise TimeoutError("Window did not move/resize to the expected position in time.")

    def pixel_matches(self, x: int, y: int, color: tuple[int, int, int], tolerance = 0):
        pix = pixel(*self.client_to_screen(x, y))

        for i in range(3):  # RGB
            if abs(pix[i] - color[i]) > tolerance:
                return False
        return True

    def wait_for_color(self, x: int, y: int, color: tuple[int, int, int], tolerance = 0, ignore_error: bool = False, timeout = 10, click: tuple[int, int] | None = None, delay: float = 0.01):
        self.ensure_window()
        start_time = time.time()
        while not self.pixel_matches(x, y, color, tolerance):
            if click:
                self.click(*click)
            time.sleep(delay)
            if time.time() - start_time > timeout:
                message = f"more than {timeout} seconds waiting for color {color} at ({x}, {y})"
                if ignore_error:
                    log(f"WARNING: {message}")
                    return
                else:
                    raise TimeoutError(message)
        time.sleep(0.05)
    
    def capture(self):
        self.ensure_window()
        left, top, right, bottom = win32gui.GetClientRect(self.hwnd)
        width = right - left
        height = bottom - top

        screen_left, screen_top = self.client_to_screen(left, top)
        region = (screen_left, screen_top, width, height)
        # Clip region to stay within screen bounds
        screen_width, screen_height = pyautogui.size()
        region = (
            max(0, min(screen_left, screen_width)),
            max(0, min(screen_top, screen_height)), 
            min(width, screen_width - max(0, screen_left)),
            min(height, screen_height - max(0, screen_top))
        )
        img = screenshot(region=region)
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def extract_first_trainer_id_char(self):
        screenshot = self.capture()
        screenshot = crop(screenshot, (224, 61, 22, 29))
        mask = extract_brown(screenshot)
        hash = hasher.compute(mask)
        return digest_hash(hash)
    
    def has_trainer_id(self, retry = True):
        ZERO_HASH = '0bfed2febcef2cff'
        T_HASH = '33cfcfefcc74f474'
        self.click(394, 78)
        hash = self.extract_first_trainer_id_char()

        if compare_hashes(hash, ZERO_HASH) < 5:
            return False
        
        if retry and compare_hashes(hash, T_HASH) < 5:
            return self.has_trainer_id(retry=False)

        return True

class RerollAgent(Agent):
    def __init__(self, window_title: str, config: dict):
        super().__init__(window_title)
        self.config = config
        self.target = Counter(config['target'])
        self.rolls = Counter()
    
    def wait_for_white_screen(self, timeout = 20, click: tuple[int, int] | None = None):
        end_time = time.time() + timeout
        while time.time() < end_time:
            screenshot = self.capture()
            sum = np.sum(screenshot > 250)
            ratio = sum / screenshot.size
            if ratio > 0.8:
                return
            if click:
                self.click(*click)
        raise TimeoutError("White screen not found")

    def delete_account(self):
        self.rolls.clear()
        # wait until menu title bar is visible
        self.wait_for_color(948, 36, (144, 213, 8), click=(1823, 970)) # click hamburger in the meantime
        time.sleep(0.6) 
        self.click(953, 730) # click delete account
        time.sleep(0.6)
        # wait for white screen by checking screenshot
        self.set_window_ensurance(False)
        self.wait_for_white_screen(click=(954, 677))
        self.set_window_ensurance(True)
        self.ensure_window()
        
    def accept_terms(self):
        self.click(1155, 455) # view terms
        self.click(1155, 563) # view privacy policy
        self.click(1089, 734) # click I agree

    def select_country(self):
        self.click(1144, 539) # click change
        time.sleep(0.3) # wait for popup
        self.click(1085, 745) # click ok popup
        time.sleep(0.3) # wait for popup close
        self.standard_ok()

    def confirm_age(self):
        self.click(956, 538) # click input
        pyautogui.typewrite(self.config['age'])
        self.standard_ok()

    def trainer_info(self):
        self.click(999, 469) # click input
        pyautogui.typewrite(self.config['name'])
        self.click(832, 550) # click male
        self.click(952, 679) # click register

    def standard_ok(self):
        self.click(1085, 677) # click ok

    def register(self):
        # wait for button shadow
        self.wait_for_color(1068, 778, (182, 181, 188), click=(960, 540)) # tap to start in the meantime
        self.accept_terms()
        time.sleep(0.6)
        self.select_country()
        time.sleep(0.6)
        self.confirm_age()

        time.sleep(0.2)
        self.wait_for_color(1169, 679, (137, 206, 0)) # button green
        # self.wait_for_color(1169, 679, (131, 200, 0)) # button green
        time.sleep(0.6)
        self.standard_ok()
        time.sleep(0.6)

        self.trainer_info()
        self.wait_for_color(901, 512, (236, 231, 228)) # gray label
        time.sleep(0.4) # wait for animation
        self.standard_ok() # confirm info

    def skip_news(self):
        # wait for notification
        self.wait_for_color(788, 38, (255, 51, 118), click=(896, 987), timeout=40) # click skip in the meantime

    def claim_present(self):
        self.click(792, 741) # click present
        self.wait_for_color(702, 994, (185, 183, 196)) # button shadow
        time.sleep(0.4)
        self.click(680, 963) # click collect all

    def select_banner(self):
        time.sleep(0.4)
        # wait until 4th green dot is filled
        self.wait_for_color(548, 716, (156, 223, 24), click=(857, 633), delay=0.4) # click right until 4th page

    def go_to_scout(self):
        # click scout until scout title is visible
        self.wait_for_color(167, 28, (247, 66, 165), click=(786, 1028))

    def wait_for_gacha_skip(self):
        self.wait_for_color(897, 989, (153, 90, 41), ignore_error=True)

    def update_rolls(self, cards: Iterable[np.ndarray]):
        for support, _ in self.read_rolls(cards):
            self.rolls.update([support['url_name']])

    def scout_bulk(self, initial: bool):
        if initial:
            self.click(737, 835) # click 10x scout
            time.sleep(0.4)
        else:
            self.click(677, 975) # click scout again
            time.sleep(0.4)
        self.click(680, 680) # click scout
        self.wait_for_gacha_skip()
        # click skip button until scout again button is visible
        self.wait_for_color(677, 976, (130, 206, 8), click=(898, 989))
        self.update_rolls(self.extract_cards())

    def scout_single(self, initial: bool):
        if initial: 
            self.click(549, 835) # click scout 1x
            time.sleep(0.4)
        else:
            self.click(676, 740) # click scout again
            time.sleep(0.4)
        self.click(680, 680) # click scout
        self.wait_for_gacha_skip()
        # click skip button until scout again button is visible
        self.wait_for_color(676, 740, (130, 206, 8), click=(898, 989))
        self.update_rolls(self.extract_single_card())

    def targets_needed(self):
        return sum((self.target - self.rolls).values())

    def roll_down(self) -> bool:
        """
            Returns True if the targets are met, False otherwise.
        """
        pulls_left = self.config['pulls']
        bulk, single = divmod(pulls_left, 10)

        for i in range(bulk):
            self.scout_bulk(i == 0)
            pulls_left -= 10
            if self.targets_needed() == 0:
                return True
            if self.targets_needed() > pulls_left:
                return False

        if bulk > 0:
            self.click(420, 977) # 10x scout back button
            self.wait_for_color(167, 28, (247, 66, 165)) # scout title

        for i in range(single):
            self.scout_single(i == 0)
            pulls_left -= 1
            if self.targets_needed() == 0:
                return True
            if self.targets_needed() > pulls_left:
                return False

        return False

    def to_title_screen(self):
        self.click(1502, 870)
        

def log(message: str):
    print(f'[{time.strftime("%H:%M:%S")}] {message}')

def wait_for_window(window_title: str, end_time: float = float('inf'), visible: bool = True, ):
    while time.time() < end_time:
        windows = pygetwindow.getWindowsWithTitle(window_title)
        if windows if visible else not windows:
            break
        time.sleep(0.1)
    else:
        raise TimeoutError("Window did not appear")

def restart_game(config: dict, timeout = 10):
    WINDOW_TITLE = "Umamusume"
    default_path = r"C:\Program Files (x86)\Steam\steamapps\common\UmamusumePrettyDerby\UmamusumePrettyDerby.exe"

    window = pygetwindow.getWindowsWithTitle(WINDOW_TITLE)
    if window:
        window[0].close()
        wait_for_window(WINDOW_TITLE, visible=False)

    path = config.get('path', default_path)
    log(f'Starting game...')
    os.startfile(path)

    start_time = time.time()
    end_time = start_time + timeout

    wait_for_window(WINDOW_TITLE, end_time)
    wait_for_window(WINDOW_TITLE, end_time, visible=False)
    wait_for_window(WINDOW_TITLE, end_time)

    log(f'Elapsed time to start game: {time.time() - start_time:.2f}s')

    agent = RerollAgent(WINDOW_TITLE, config)
    # agent.wait_for_color(86, 945, (0, 92, 173), click=(1920 // 2, 1080 // 2)) # click until criware
    agent.wait_for_white_screen(click=(960, 800))
    if agent.has_trainer_id():
        agent.delete_account()
    return agent

def main():
    with open('config.json', 'r') as f:
        config = json.load(f)
    n = 1
    agent = restart_game(config)
    start_time = time.time()
    while True:
        try:
            log(f'Registering account {n}...')
            agent.register()
            agent.skip_news()
            agent.claim_present()
            agent.go_to_scout()
            agent.select_banner()
            time.sleep(0.4)
            if agent.roll_down():
                log('Targets met!')
                exit(0)
            for url_name, count in agent.rolls.items():
                if url_name in agent.target:
                    log(f'Pulled {count}x {url_name}')
            agent.to_title_screen()
            agent.delete_account()
            
            if n % 10 == 0:
                elapsed = time.time() - start_time
                avg_time = elapsed / n
                log(f'Average time per reroll: {avg_time:.2f}s')
        except Exception as e:
            traceback.print_exc()
            log(f'Error: {e}, Restarting...')
            agent = restart_game(config)
        finally:
            n += 1

if __name__ == '__main__':
    main()
