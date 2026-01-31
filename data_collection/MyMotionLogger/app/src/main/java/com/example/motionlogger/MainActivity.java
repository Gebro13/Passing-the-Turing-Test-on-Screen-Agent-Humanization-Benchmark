package com.example.motionlogger;

import android.app.Activity;
import android.content.Context;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.os.Bundle;
import android.os.SystemClock;
import android.util.Log;
import android.view.KeyEvent;
import android.view.MotionEvent;
import android.view.View;
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



    private FileOutputStream logStream;

    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        FrameLayout layout = new FrameLayout(this);
        layout.setBackgroundColor(0xFFFFFFFF); // white background
        setContentView(layout);

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

        sensorManager = (SensorManager) getSystemService(Context.SENSOR_SERVICE);
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
        sb.append(event.getEventTime()).append(" MotionEvent: action=").append(actionMasked)
                .append(" actionIndex=").append(actionIndex)
                .append(" pointerCount=").append(pointerCount).append("\n");
        for (int i = 0; i < pointerCount; i++) {
            sb.append("  pointer[").append(i).append("] id=").append(event.getPointerId(i))
                    .append(" x=").append(event.getX(i))
                    .append(" y=").append(event.getY(i))
                    .append(" pressure=").append(event.getPressure(i))
                    .append(" size=").append(event.getSize(i))
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

    @Override
    public void onSensorChanged(SensorEvent event) {
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

    @Override
    public boolean onKeyDown(int keyCode, KeyEvent event) {
        String data = String.format("%d KeyEvent DOWN: keyCode=%d (%s)\n",
                event.getEventTime(), keyCode, KeyEvent.keyCodeToString(keyCode));
        logToFile(data);
        return super.onKeyDown(keyCode, event);
    }

    @Override
    public boolean onKeyUp(int keyCode, KeyEvent event) {
        String data = String.format("%d KeyEvent UP: keyCode=%d (%s)\n",
                event.getEventTime(), keyCode, KeyEvent.keyCodeToString(keyCode));
        logToFile(data);
        return super.onKeyUp(keyCode, event);
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
