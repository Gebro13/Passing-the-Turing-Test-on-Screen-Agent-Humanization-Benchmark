from typing import Callable, List, Tuple, Optional
import time
import data_collection.automations_general_android as general



password: List[Tuple[int, int]] = [

]


def start_phone():
    # power()
    general.swipe(540, 1800, 540, 700, duration_ms=200)
    for x, y in password:
        general.tap(x, y)
        time.sleep(0.1)

def clean_every_app(final_wait_seconds: float = 4.0):
    """
    start: anywhere anystate  
    end: home screen with no app in background  
    """
    for _ in range(10):
        general.home()
        time.sleep(0.1)
    general.app_switch()
    time.sleep(0.3)
    general.tap(540, 1800)
    time.sleep(final_wait_seconds)

def prepare_apps_screen():
    """
    start: anywhere anystate  
    end: app screen with no app in background  
    """
    clean_every_app()
    # for i in range(3): # no need now because app screen is set to top 
    #     swipe(900, 1100, 100, 1100, duration_ms=300)
    # time.sleep(0.1)

def switch_app_from_sensorevent_to_appscreen():
    """
    
    start: sensor event app in foreground
    end: app screen in foreground, May Not Be Fully Loaded; it depends on the app.
    """
    time.sleep(1.0)
    general.app_switch()
    time.sleep(1.0)
    general.tap(850, 1300)
    time.sleep(1.0)


x_coords = [160, 420, 680, 940]
y_coords = [150,
            390,
            630,
            870,
            1110, 
            1350]

table_of_app_funcs = [
["",        "zhihu",   "voov",     "qqmusic"],
["tdocs",   "taobao",  "qunar",    "jd"     ],
["",        "",        "iqiyi",    "ctrip"  ],
["eleme",   "meituan", "bilibili", "rednote"],
["gaode",   "haodf",   "toutiao",  "youdao"],
["cainiao", "weibo",   "dianping", "umetrip"]
]

def query_position(app_func_name: str) -> Tuple[int, int]:
    for (i, row) in enumerate(table_of_app_funcs):
        for (j, func_name) in enumerate(row):
            if func_name == app_func_name:
                return (x_coords[j], y_coords[i])
    raise ValueError(f"App function name {app_func_name} not found in table.")

def jd():
    general.tap(916, 373)
    time.sleep(7.0)
    general.tap(665, 243)

def ctrip():
    general.tap(898, 646)
    time.sleep(7.0)

def ctrip_disturbance_resolution():
    if not general.pixel_on_screenshot_is_color(540, 45, (37, 130, 245), 10):
        general.swipe(540, 266, 540, 1412, duration_ms=2000)

def eleme():
    general.tap(132, 843) # open eleme
    time.sleep(7.0)
    general.tap(210, 927) # select sjtu as the place
    time.sleep(1.0)

def eleme_cleanup():
    prepare_apps_screen()
    eleme()
    general.tap(754, 1872) # click for the cart
    time.sleep(0.5)
    while True:
        if general.pixel_on_screenshot_is_color(720, 650, (214,229,231), 10):
            # 720, 650 # place for a gray-blue shadow
            # 470, 1223 # place for a very special blue icon
            break
        bin_coordinates = (991, 402)
        if general.pixel_on_screenshot_is_color(*bin_coordinates, (254, 246, 223), 10):
            print("The bin icon is not in the expected place.")
            bin_coordinates = (991, 524) # try a lower place
        general.tap(*bin_coordinates) # click delete for the first shop
        time.sleep(0.2)
        general.tap(709, 1090) # confirm delete
        time.sleep(1.0)

def qqmusic():
    general.tap(910, 156)
    time.sleep(15.0)

def iqiyi():
    general.tap(682, 626)
    time.sleep(7.0)

def tdocs():
    general.tap(160, 390)
    time.sleep(10.0)
    general.tap(860, 1530)
    time.sleep(1.0)
    general.tap(860, 1530)
    time.sleep(1.0)


"""
def general_run_app(app_name: str):
    # first check whether app_name is defined in global space; if not, query its position
    if app_name in globals():
        globals()[app_name]()
    else:
        pos = query_position(app_name)
        print(f"General run app: tapping at position {pos} for app {app_name}")
        tap(*pos)
        time.sleep(15.0)
"""

def get_app_general_launcher(app_name: str) -> Callable[[], None]:
    """
        Naive click and wait.  
        No constant overhead before clicking the app icon.
    """
    def func():
        pos = query_position(app_name)
        general.tap(*pos)
        time.sleep(15.0)
    return func


def general_get_app_func(app_name: str) -> Callable[[], None]:
    """
        If the app has a specifically defined function, return that function, which may have some specific preparations for that app. Otherwise, return a general launcher function that just taps on the app icon and waits.
    """
    if app_name in globals():
        return globals()[app_name]
    else:
        return get_app_general_launcher(app_name)