package com.example.motionlogger;

import android.app.Activity;
import android.content.Context;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.media.AudioManager;
import android.media.ToneGenerator;
import android.os.Bundle;
import android.os.SystemClock;
import android.text.Editable;
import android.text.TextWatcher;
import android.util.Log;
import android.view.KeyEvent;
import android.view.MotionEvent;
import android.view.View;
import android.view.inputmethod.EditorInfo;
import android.view.inputmethod.InputConnection;
import android.view.inputmethod.InputConnectionWrapper;
import android.widget.FrameLayout;

import androidx.annotation.Nullable;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Locale;

public class MainActivity extends Activity implements SensorEventListener {
    private static final String TAG = "MotionLogger";
    private SensorManager sensorManager;
    private Sensor gyroscope;
    private Sensor accelerometer;
    private Sensor rotationVector;
    private Sensor magneticField;
    private Sensor motionDetector;
    private Sensor light;
    private Sensor proximity;
    private Sensor pressure;
    private Sensor gravity;
    private Sensor linearAcceleration;
    private Sensor stepCounter;
    private Sensor stepDetector;
    private boolean beeping = false;


    public class LoggingEditText extends androidx.appcompat.widget.AppCompatEditText {
        public LoggingEditText(Context context) {
            super(context);
        }

        @Override
        public InputConnection onCreateInputConnection(EditorInfo outAttrs) {
            return new LoggingInputConnection(super.onCreateInputConnection(outAttrs), true);
        }

        private class LoggingInputConnection extends InputConnectionWrapper {
            public LoggingInputConnection(InputConnection target, boolean mutable) {
                super(target, mutable);
            }

            @Override
            public boolean commitText(CharSequence text, int newCursorPosition) {
                logToFile(SystemClock.elapsedRealtimeNanos() + " IME Commit: \"" + text + "\"\n");
                return super.commitText(text, newCursorPosition);
            }
        }
    }

    private LoggingEditText inputBox;

    private FileOutputStream logStream;

    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        FrameLayout layout = new FrameLayout(this);
        layout.setBackgroundColor(0xFFFFFFFF); // white background

        // Create the input box
        inputBox = new LoggingEditText(this);
        inputBox.setHint("Type here...");
        inputBox.setFocusable(true);
        inputBox.setFocusableInTouchMode(true);


        // Add to layout
        layout.addView(inputBox, new FrameLayout.LayoutParams(
                FrameLayout.LayoutParams.MATCH_PARENT,
                FrameLayout.LayoutParams.WRAP_CONTENT
        ));

        setContentView(layout);

        inputBox.requestFocus(); // may need to remove this line

        // get even more raw events (including multiple key presses)
        // Add this in onCreate after setting up the inputBox
        inputBox.setOnKeyListener(new View.OnKeyListener() {
            @Override  // a KeyEventListener
            public boolean onKey(View v, int keyCode, KeyEvent event) {
                // This will catch events before they're processed by the EditText
                String data = String.format("%d InputBox Key: action=%d keyCode=%d (%s) chars=%s\n",
                        event.getEventTime(),
                        event.getAction(),
                        keyCode,
                        KeyEvent.keyCodeToString(keyCode),
                        event.getCharacters());
                logToFile(data);

                // Return false to let the event continue to the EditText
                return false;
            }
        });

        // a KeyEventListener
        // Add this in onCreate after setting up the inputBox
        inputBox.addTextChangedListener(new TextWatcher() {
            @Override
            public void beforeTextChanged(CharSequence s, int start, int count, int after) {
                logToFile(SystemClock.elapsedRealtimeNanos() + " Text before change: \"" + s + "\"\n");
            }

            @Override
            public void onTextChanged(CharSequence s, int start, int before, int count) {
                logToFile(SystemClock.elapsedRealtimeNanos() + " Text changing: \"" + s + "\"\n");
            }

            @Override
            public void afterTextChanged(Editable s) {
                logToFile(SystemClock.elapsedRealtimeNanos() + " Text after change: \"" + s + "\"\n");
            }
        });

        try {
            File logFile = new File(getExternalFilesDir(null), "motion_log.txt");
            logStream = new FileOutputStream(logFile, false);
        } catch (IOException e) {
            Log.e(TAG, "Failed to open log file", e);
        }

        layout.setOnTouchListener(new View.OnTouchListener() {
            @Override
            public boolean onTouch(View v, MotionEvent event) {
                String logEntry = formatMotionEvent(event);
                logToFile(logEntry);
                return true;
            }
        });

        boolean noSensors = getIntent().getBooleanExtra("no_sensors", false);
        beeping = getIntent().getBooleanExtra("beeping", false);

        sensorManager = (SensorManager) getSystemService(Context.SENSOR_SERVICE);

        if (!noSensors) {
            // print the bool value
            // logToFile(String.format("%d no_sensors=%b\n", SystemClock.elapsedRealtimeNanos(), noSensors));
            // print the !bool value
            // logToFile(String.format("%d yes_sensors=%b\n", SystemClock.elapsedRealtimeNanos(), !noSensors));
            // logToFile(String.format("%d Sensor logging started\n", SystemClock.elapsedRealtimeNanos()));

        accelerometer = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER);
        gyroscope = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE);
        magneticField = sensorManager.getDefaultSensor(Sensor.TYPE_MAGNETIC_FIELD);
        motionDetector = sensorManager.getDefaultSensor(Sensor.TYPE_MOTION_DETECT);
        light = sensorManager.getDefaultSensor(Sensor.TYPE_LIGHT);
        proximity = sensorManager.getDefaultSensor(Sensor.TYPE_PROXIMITY);
        pressure = sensorManager.getDefaultSensor(Sensor.TYPE_PRESSURE);
        gravity = sensorManager.getDefaultSensor(Sensor.TYPE_GRAVITY);
        linearAcceleration = sensorManager.getDefaultSensor(Sensor.TYPE_LINEAR_ACCELERATION);
        rotationVector = sensorManager.getDefaultSensor(Sensor.TYPE_ROTATION_VECTOR);
        stepCounter = sensorManager.getDefaultSensor(Sensor.TYPE_STEP_COUNTER);
        stepDetector = sensorManager.getDefaultSensor(Sensor.TYPE_STEP_DETECTOR);

        if (gyroscope != null) {
            sensorManager.registerListener(this, gyroscope, SensorManager.SENSOR_DELAY_FASTEST);
        }
        if (accelerometer != null) {
            sensorManager.registerListener(this, accelerometer, SensorManager.SENSOR_DELAY_FASTEST);
        }
        if (rotationVector != null) {
            sensorManager.registerListener(this, rotationVector, SensorManager.SENSOR_DELAY_FASTEST);
        }
        if (magneticField != null) {
            sensorManager.registerListener(this, magneticField, SensorManager.SENSOR_DELAY_FASTEST);
        }
        if (motionDetector != null) {
            sensorManager.registerListener(this, motionDetector, SensorManager.SENSOR_DELAY_FASTEST);
        }
        if (light != null) {
            sensorManager.registerListener(this, light, SensorManager.SENSOR_DELAY_FASTEST);
        }
        if (proximity != null) {
            sensorManager.registerListener(this, proximity, SensorManager.SENSOR_DELAY_FASTEST);
        }
        if (pressure != null) {
            sensorManager.registerListener(this, pressure, SensorManager.SENSOR_DELAY_FASTEST);
        }
        if (gravity != null) {
            sensorManager.registerListener(this, gravity, SensorManager.SENSOR_DELAY_FASTEST);
        }
        if (linearAcceleration != null) {
            sensorManager.registerListener(this, linearAcceleration, SensorManager.SENSOR_DELAY_FASTEST);
        }
        if (stepCounter != null) {
            sensorManager.registerListener(this, stepCounter, SensorManager.SENSOR_DELAY_FASTEST);
        }
        if (stepDetector != null) {
            sensorManager.registerListener(this, stepDetector, SensorManager.SENSOR_DELAY_FASTEST);
        }
        }


        long resulting_offset = sampleBootOffsetNs(10);
        String offset_string = String.format("%d\n", resulting_offset);
        logToFile(offset_string);
    }

    public static long sampleBootOffsetNs(int trials) {
        // o4-mini code that claims to "get the best..."
        long bestOffset = Long.MAX_VALUE;
        long bestDelta = Long.MAX_VALUE;
        for (int i = 0; i < trials; i++) {
            long t1 = SystemClock.elapsedRealtimeNanos();
            long up = SystemClock.uptimeMillis();
            long t2 = SystemClock.elapsedRealtimeNanos();
            long delta = t2 - t1;
            // use the midpoint to minimize call-ordering error
            long mid = t1 + delta/2;
            long offset = mid - up*1_000_000L;
            if (delta < bestDelta) {
                bestDelta = delta;
                bestOffset = offset;
            }
        }
        return bestOffset;
    }

    private String formatMotionEvent(MotionEvent event) {
        StringBuilder sb = new StringBuilder();
        int pointerCount = event.getPointerCount();
        int actionMasked = event.getActionMasked();
        int actionIndex = event.getActionIndex();
        int historySize = event.getHistorySize();
        // getRawX()/getRawY() (no-arg, pointer 0) available since API 1.
        // The raw-to-view offset is a constant translation across all pointers/history.
        float rawOffsetX = event.getRawX() - event.getX();
        float rawOffsetY = event.getRawY() - event.getY();
        sb.append(event.getEventTime()).append(" MotionEvent: action=").append(actionMasked)
                .append(" actionIndex=").append(actionIndex)
                .append(" pointerCount=").append(pointerCount)
                .append(" historySize=").append(historySize)
                .append(" downTime=").append(event.getDownTime())
                .append(" rawAction=").append(event.getAction())
                .append(" source=").append(event.getSource())
                .append(" deviceId=").append(event.getDeviceId())
                .append(" flags=").append(event.getFlags())
                .append(" edgeFlags=").append(event.getEdgeFlags())
                .append(" metaState=").append(event.getMetaState())
                .append(" buttonState=").append(event.getButtonState())
                // .append(" classification=").append(event.getClassification()) # not available in api 25
                .append(" xPrecision=").append(event.getXPrecision())
                .append(" yPrecision=").append(event.getYPrecision())
                .append(" rawX=").append(event.getRawX())
                .append(" rawY=").append(event.getRawY())
                .append(" x=").append(event.getX())
                .append(" y=").append(event.getY())
                .append("\n");
        for (int h = 0; h < historySize; h++) {
            sb.append("  historical[").append(h).append("] time=").append(event.getHistoricalEventTime(h)).append("\n");
            for (int i = 0; i < pointerCount; i++) {
                float hx = event.getHistoricalX(i, h);
                float hy = event.getHistoricalY(i, h);
                sb.append("    pointer[").append(i).append("] id=").append(event.getPointerId(i))
                        .append(" x=").append(hx)
                        .append(" y=").append(hy)
                        .append(" deductedRawX=").append(hx + rawOffsetX)
                        .append(" deductedRawY=").append(hy + rawOffsetY)
                        .append(" axisvalueX=").append(event.getHistoricalAxisValue(MotionEvent.AXIS_X, i, h))
                        .append(" axisvalueY=").append(event.getHistoricalAxisValue(MotionEvent.AXIS_Y, i, h))
                        .append(" pressure=").append(event.getHistoricalPressure(i, h))
                        .append(" orientation=").append(event.getHistoricalOrientation(i, h))
                        .append(" size=").append(event.getHistoricalSize(i, h))
                        .append(" touchMajor=").append(event.getHistoricalTouchMajor(i, h))
                        .append(" touchMinor=").append(event.getHistoricalTouchMinor(i, h))
                        .append(" toolMajor=").append(event.getHistoricalToolMajor(i, h))
                        .append(" toolMinor=").append(event.getHistoricalToolMinor(i, h))
                        .append(" tilt=").append(event.getHistoricalAxisValue(MotionEvent.AXIS_TILT, i, h))
                        .append(" distance=").append(event.getHistoricalAxisValue(MotionEvent.AXIS_DISTANCE, i, h))
                        .append(" toolType=").append(event.getToolType(i))
                        .append("\n");
            }
        }
        sb.append("  current: time=").append(event.getEventTime()).append("\n");
        for (int i = 0; i < pointerCount; i++) {
            float cx = event.getX(i);
            float cy = event.getY(i);
            sb.append("    pointer[").append(i).append("] id=").append(event.getPointerId(i))
                    .append(" x=").append(cx)
                    .append(" y=").append(cy)
                    .append(" deductedRawX=").append(cx + rawOffsetX)
                    .append(" deductedRawY=").append(cy + rawOffsetY)
                    .append(" axisvalueX=").append(event.getAxisValue(MotionEvent.AXIS_X, i))
                    .append(" axisvalueY=").append(event.getAxisValue(MotionEvent.AXIS_Y, i))
                    .append(" pressure=").append(event.getPressure(i))
                    .append(" orientation=").append(event.getOrientation(i))
                    .append(" size=").append(event.getSize(i))
                    .append(" touchMajor=").append(event.getTouchMajor(i))
                    .append(" touchMinor=").append(event.getTouchMinor(i))
                    .append(" toolMajor=").append(event.getToolMajor(i))
                    .append(" toolMinor=").append(event.getToolMinor(i))
                    .append(" tilt=").append(event.getAxisValue(MotionEvent.AXIS_TILT, i))
                    .append(" distance=").append(event.getAxisValue(MotionEvent.AXIS_DISTANCE, i))
                    .append(" toolType=").append(event.getToolType(i))
                    .append("\n");
        }
        return sb.toString();
    }

    private void logToFile(String data) {
        try {
            logStream.write(data.getBytes());
            logStream.flush();
        } catch (IOException e) {
            Log.e(TAG, "Write failed", e);
        }
    }

    private String timestamp() {
        return new SimpleDateFormat("yyyy-MM-dd HH:mm:ss.SSS", Locale.US).format(new Date());
    }

    private void debugBeep() {
        try {
            ToneGenerator toneGen = new ToneGenerator(AudioManager.STREAM_NOTIFICATION, 100);
            toneGen.startTone(ToneGenerator.TONE_PROP_BEEP, 200);
        } catch (Exception e) {
            Log.e(TAG, "Beep failed", e);
        }
    }

    private long lastBeepTime = 0;

    @Override
    public void onSensorChanged(SensorEvent event) {

        if (beeping) {
            long currentTime = SystemClock.elapsedRealtime();
            if (currentTime - lastBeepTime > 10000) { // 10 seconds in milliseconds
                lastBeepTime = currentTime;
                debugBeep();
            }
        }

        String sensorType;
        switch (event.sensor.getType()) {
            case Sensor.TYPE_ACCELEROMETER:
                sensorType = "Accelerometer";
                break;
            case Sensor.TYPE_GRAVITY:
                sensorType = "Gravity";
                break;
            case Sensor.TYPE_GYROSCOPE:
                sensorType = "Gyroscope";
                break;
            case Sensor.TYPE_LIGHT:
                sensorType = "Light";
                break;
            case Sensor.TYPE_LINEAR_ACCELERATION:
                sensorType = "LinearAcceleration";
                break;
            case Sensor.TYPE_MAGNETIC_FIELD:
                sensorType = "MagneticField";
                break;
            case Sensor.TYPE_MOTION_DETECT:
                sensorType = "MotionDetector";
                break;
            case Sensor.TYPE_PRESSURE:
                sensorType = "Pressure";
                break;
            case Sensor.TYPE_PROXIMITY:
                sensorType = "Proximity";
                break;
            case Sensor.TYPE_ROTATION_VECTOR:
                sensorType = "RotationVector";
                break;
            case Sensor.TYPE_STEP_COUNTER:
                sensorType = "StepCounter";
                break;
            case Sensor.TYPE_STEP_DETECTOR:
                sensorType = "StepDetector";
                break;
            default:
                return;
        }
        String data;
        switch (event.sensor.getType()) {
            // see Sensor.sSensorReportingModes or https://developer.android.com/reference/android/hardware/SensorEvent?hl=en#values
            // 1 values
            case Sensor.TYPE_LIGHT:
            case Sensor.TYPE_MOTION_DETECT:
            case Sensor.TYPE_PRESSURE:
            case Sensor.TYPE_PROXIMITY:
            case Sensor.TYPE_STEP_COUNTER:
            case Sensor.TYPE_STEP_DETECTOR:
                data = String.format(Locale.US, "%d %s: value=%f\n",
                        event.timestamp, sensorType, event.values[0]);
                break;

            // 3 values
            case Sensor.TYPE_ACCELEROMETER:
            case Sensor.TYPE_GYROSCOPE:
            case Sensor.TYPE_GRAVITY:
            case Sensor.TYPE_LINEAR_ACCELERATION:
            case Sensor.TYPE_MAGNETIC_FIELD:
                data = String.format(Locale.US, "%d %s: x=%f y=%f z=%f\n",
                    event.timestamp, sensorType, event.values[0], event.values[1], event.values[2]);
                break;

            // https://www.zhihu.com/tardis/zm/art/97186723?source_id=1005
            case Sensor.TYPE_ROTATION_VECTOR:
                data = String.format(Locale.US, "%d %s: i=%f, j=%f, k=%f, w=%f, accuracy=%f\n",
                        event.timestamp, sensorType, event.values[0], event.values[1], event.values[2], event.values[3], event.values[4]);
                break;
            default:
                return;
        }
        logToFile(data);
    }

    @Override
    public void onAccuracyChanged(Sensor sensor, int accuracy) {
        // Optional: log accuracy changes if needed
    }

    @Override // a KeyEventListener
    public boolean onKeyDown(int keyCode, KeyEvent event) {
        String data = String.format("%d KeyEvent DOWN: keyCode=%d (%s) repeat=%d meta=%d\n",
                event.getEventTime(),
                keyCode,
                KeyEvent.keyCodeToString(keyCode),
                event.getRepeatCount(),
                event.getMetaState());
        logToFile(data);

        // Let the system handle it normally too
        return super.onKeyDown(keyCode, event);
    }

    @Override // a KeyEventListener
    public boolean onKeyUp(int keyCode, KeyEvent event) {
        String data = String.format("%d KeyEvent UP: keyCode=%d (%s) meta=%d\n",
                event.getEventTime(),
                keyCode,
                KeyEvent.keyCodeToString(keyCode),
                event.getMetaState());
        logToFile(data);

        // Let the system handle it normally too
        return super.onKeyUp(keyCode, event);
    }

    @Override  // a KeyEventListener
    public boolean dispatchKeyEvent(KeyEvent event) {
        String data = String.format("%d Dispatch Key: action=%d keyCode=%d (%s) repeat=%d\n",
                event.getEventTime(),
                event.getAction(),
                event.getKeyCode(),
                KeyEvent.keyCodeToString(event.getKeyCode()),
                event.getRepeatCount());
        logToFile(data);

        return super.dispatchKeyEvent(event);
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        sensorManager.unregisterListener(this);
        try {
            if (logStream != null) logStream.close();
        } catch (IOException e) {
            Log.e(TAG, "Error closing file", e);
        }
    }
}
