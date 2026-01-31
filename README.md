# Agent Humanization Paradigm

This is the repository for collecting data from android and parsing them.

# Contents

- [1. Data collection](#1-data-collection)
    - [Test device availability](#test-device-availability)
        - [Debugging permissions](#debugging-permissions)
        - [`getevent`](#getevent)
        - [`screenrecord`](#screenrecord)
        - [Launching an apk with `adb`](#launching-an-apk-with-adb)
    - [Start collecting data](#start-collecting-data)
- [2. Parsing Data](#2-parsing-data)
    - [Primitive python test](#primitive-python-test)


# 1. Data collection

Dependency: `adb`, `python` on your linux desktop and an android device

## Test device availability

Please run these commands line by line at first; if some of them cannot run, the error won't show up when you run the python script.

### debugging permissions

```bash
adb devices
```
should show a list of device(s) without warning.

### `getevent`

type
```bash
adb shell -t -t getevent -lt
```
and touch your android's screen; should show some logs being printed out. Press `Ctrl+C` to stop.

### `screenrecord`

#### availability
May not be available in some legacy devices.
type 
```bash
adb shell screenrecord /sdcard/example1.mp4
```
; after a while, press `Ctrl+C` to stop the recording. Then
```bash
adb pull /sdcard/example1.mp4 example1.mp4
```
to pull it into your current folder.

Also, you can use 
```bash
adb shell pkill -l SIGINT screenrecord
```
to halt the process without focusing on the terminal.

### Soft Keyboards

you need to install https://github.com/senzhk/ADBKeyBoard to your android device.

Setting the keyboard to it from bash: 
```bash
adb shell ime set com.android.adbkeyboard/.AdbIME
```

#### (Optional) Extend maximum recording time to beyond 3 minutes

If you want to keep using an android-native command to do screenrecord, but want to extend the maximum recording time to be larger than 180s, you can:
1. pull the `screenrecord` binary to your PC
```bash
adb pull /system/bin/screenrecord screenrecord
```
2. `xxd screenrecord > screenrecord.hex` to create a hex dump of the binary
3. modify the hex dump to change the maximum recording time
    - For my device, modify the last occurrence of `b400 0000` (int32 180) (3 minutes)
    - You can modify it into `0807 0000` (1800). We did this, and thus our logging limit is 30min.
4. `xxd -r screenrecord.hex > screenrecord_modified` to create a modified binary
5. `chmod +x screenrecord_modified`
6. push the modified binary back to the device.
    - To use existing `controller.py`, you need to put the binary under `/data/local/tmp/`.

Note: 
- for this method, the default recording time will be longer, but the restriction on `--time-limit` will still be \<=180s. Therefore, you shouldn't use this argument.
- The process halting command is the sameas before; do not halt `screenrecord_modified`.

### launching our motion logger with `adb`

#### availability

Requirements: `android-studio`(if you want to compile from source), `adb`.

**Compiling from source** Pull the java branch of this repository in another directory; then compile the apk into the phone using `android-studio`.  
**Direct installation** use the compiled apk.

#### Functionality

Existing loggers are set to the highest sampling frequency. You can easily add other sensors and tune the sampling frequency.

Get the activity to launch:
```bash
adb shell pm dump com.example.motionlogger | grep -A 1 "MAIN"
```
which yields `com.example.motionlogger/.MainActivity`.
Then
```bash
adb shell am start -S com.example.motionlogger/.MainActivity
```
where `-S` stops any previous running instances.
Then
```bash
adb shell am force-stop com.example.motionlogger/.MainActivity
```



#### (optional) Making the apk not sleep and record nothing after a while(typically 3 minutes)

change the allowed number of parallel background activities in the system settings' developer options.  
Also, add the app as debug mode in developer options.


### (optional) `getevent`-visible agent swipe

Pull https://github.com/Cartucho/android-touch-record-replay and follow their instructions until you succeed(you probably have to change the binary `mysendevent` to `mysendevent-arm64`).  
If you follow their instructions, you should end up with a binary under `/data/local/tmp/` in your phone.   
Then switch back to this repo, and:

```python
import fake_adb.adb_wrapper as controller_fake_adb
controller_fake_adb.MotionGenerator.swipe(adb_path="adb", x1=400, y1=1300, x2=800, y2=800, duration_ms=500, evdev="/dev/input/event4")
```
generates a swipe programmed with 11000us between each dot but has in fact a $600\pm 200$ us more latency. The number 11000us depends on your device.   
This may be tolerable because human swipes have also about 11000us between each dot and there is inherent noise over the interval, and the distribution can be shifted accordingly.  
In comparison, there would have been no preemptive workaround when the base latency was 0.3s per `sendevent`, as in the 6a1315c commit.

## Start collecting data

Warning: due to that timestamp alignment is tricky between motionevent and sensorevent, it is better that don't sleep your phone before collecting data. However, the first line of sensor data contains an offset in ns.
no need to launch the individual collectors.

Collecting human data:
```bash
python main.py --automatic_exit_app_and_reset --automatic_switch_app --user <user_name_just_specify_in_the_csv> --task_provide_file tasks.csv
```
Press enter to stop, don't press ctrl+c.

Collecting agent data:
```bash
python automations.py
```
no need to launch the individual collectors.

## (Optional) compressing the recorded video

```bash
bash convert_to_h265.sh --time_stamp 20250903_181934
```
