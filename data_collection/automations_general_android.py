# Provide general automations for Android devices using ADB commands, such as 
# swiping, tapping, taking screenshots, and analyzing pixel colors. 
# These functions can be used in various data collection tasks to interact
# with the device and gather information about the screen state.
# They are not recorded by the gesture event recorder so they do not interfere 
# with the gesture data collection. 

import os
from typing import Optional, Tuple
from pathlib import Path
import PIL.Image
import datetime


def tap(x: int, y: int):
    os.system(f"adb shell input tap {x} {y}")

def swipe(x1: int, y1: int, x2: int, y2: int, duration_ms: int = 300):
    os.system(f"adb shell input swipe {x1} {y1} {x2} {y2} {duration_ms}")

def power():
    os.system("adb shell input keyevent 26")

def app_switch():
    os.system("adb shell input keyevent KEYCODE_APP_SWITCH")

def home():
    os.system("adb shell input keyevent KEYCODE_HOME")

def back():
    os.system("adb shell input keyevent KEYCODE_BACK")


def fast_screenshot(adb_path: str = "adb", save_path: Optional[Path] = None) -> PIL.Image.Image:
    
    delete_after_image_open = False
    if save_path is None:
        # save to temporary folder and delete after opening the image
        save_path = Path("tmp") / f"screenshot_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        delete_after_image_open = True
    
    if True:
        command = adb_path + " shell rm /sdcard/screenshot.png"
        os.system(command)
        command = adb_path + " shell screencap -p /sdcard/screenshot.png"
        os.system(command)
        
        # if ./screenshot is not existent, then create the directory
        image_save_path_folder = save_path.parent
        if not os.path.exists(image_save_path_folder):
            os.makedirs(image_save_path_folder)

        command = adb_path + f" pull /sdcard/screenshot.png {save_path}"
        os.system(command)
    else:
        pass
    image = PIL.Image.open(save_path)
    # resize image to 1080x1920
    image = image.resize((1080, 1920))
    if delete_after_image_open:
        os.remove(save_path)
    return image

def get_rgb(image: PIL.Image.Image, x: int, y: int) -> Tuple[int, int, int]:
    pixel = image.getpixel((x, y))
    if pixel is None:
        raise ValueError("Pixel at the given coordinates is None")
    elif isinstance(pixel, int):
        return (pixel, pixel, pixel)
    elif isinstance(pixel, float):
        raise NotImplementedError("Float pixel value is not supported or not tested. What does this mean anyway.")
        val = int(pixel)
        return (val, val, val)
    elif len(pixel) == 4:
        return pixel[:3]
    elif len(pixel) == 3:
        return pixel
    else:
        raise ValueError("Unsupported pixel format")

def get_screenshot_and_get_rgb(x: int, y: int) -> Tuple[int, int, int]:
    image = fast_screenshot()
    return get_rgb(image, x, y)

def l1_distance(c1: Tuple[int, int, int], c2: Tuple[int, int, int]) -> int:
    return sum(abs(c1[i] - c2[i]) for i in range(3))

def pixel_on_screenshot_is_color(x: int, y: int, target_color: Tuple[int, int, int], l1_distance_threshold: int) -> bool:
    current_color = get_screenshot_and_get_rgb(x, y)
    distance = l1_distance(current_color, target_color)
    return distance <= l1_distance_threshold