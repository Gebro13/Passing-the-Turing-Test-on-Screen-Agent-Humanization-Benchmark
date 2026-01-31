# Java Logger

This is a logger; it can record finger/button activities when in foreground. 
It can also record accelerometer, rotation and gyroscope when in foreground/background(note: this may not be possible if api>=28)

# Dependency

- Android Studio; I installed it as a side effect using link https://appium.io/docs/en/2.19/quickstart/install/ link
- an Android phone; read off its api version thru Android Studio.


# Usage

1. Create an "Empty Views Activity" with Android Studio, commit with the default `.gitignore`.
2. Pull this branch.
3. Change the minSdk and targetSdk in `app/build.gradle.kts` to your phone's api version.
4. Click run in Android Studio to install the app.  
   If you launch only this apk:  
   5. When collecting data, keep the app in foreground. Close the app after finishing collection.  
   6. The file is on the phone and under `/sdcard/Android/data/com.example.motionlogger/Files/motion_log.txt`.  
   If you launch in synergy with other data collectors:  
   5. see branch main.  

# Interpretation of Data

The first line of the `txt` file is a positive integer in nanoseconds, the min-statistic of ((averaged nanosecond include-deep-sleep time) minus (millisecond exclude-sleep time since boot)).  
You can increase the precision to all-nanoseconds if you have api>=35.

The timestamps of `SensorEvent`s are in nanoseconds( $10^{-9}$ s) while the timestamps of `MotionEvent`s and `KeyEvent`s are in milliseconds( $10^{-3}$ s).

`SensorEvent`s and `KeyEvent`s always occupy one line while `MotionEvent`s may occupy many lines.