package com.fallguard.ai.sensor

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import com.fallguard.ai.model.RawSensorReading
import java.util.concurrent.ConcurrentLinkedDeque

/**
 * High-frequency hardware sensor collector (50 Hz).
 * Collects 3-axis accelerometer and 3-axis gyroscope telemetry.
 * Strictly maintains an in-memory bounded 5-second buffer (max 250 samples)
 * to ensure zero permanent storage of raw physical sensor data.
 */
class SensorCollector(
    private val context: Context,
    private val maxBufferSamples: Int = 250 // ~5 seconds at 50 Hz
) : SensorEventListener {

    private val sensorManager = context.getSystemService(Context.SENSOR_SERVICE) as SensorManager
    private val accelerometer = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
    private val gyroscope = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE)

    // Ephemeral rolling buffer: bounded strictly to 5 seconds
    val rollingBuffer = ConcurrentLinkedDeque<RawSensorReading>()

    // Latest readings cache
    @Volatile var latestAx = 0f; private set
    @Volatile var latestAy = 0f; private set
    @Volatile var latestAz = 9.81f; private set
    @Volatile var latestGx = 0f; private set
    @Volatile var latestGy = 0f; private set
    @Volatile var latestGz = 0f; private set

    var onReadingListener: ((RawSensorReading) -> Unit)? = null
    var isCollecting = false
        private set

    fun start() {
        if (isCollecting) return
        rollingBuffer.clear()

        // SENSOR_DELAY_GAME = 20,000 microseconds = 50 Hz
        accelerometer?.let {
            sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_GAME)
        }
        gyroscope?.let {
            sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_GAME)
        }
        isCollecting = true
    }

    fun stop() {
        if (!isCollecting) return
        sensorManager.unregisterListener(this)
        rollingBuffer.clear()
        isCollecting = false
    }

    override fun onSensorChanged(event: SensorEvent?) {
        if (event == null) return

        val nowSeconds = System.currentTimeMillis() / 1000.0

        if (event.sensor.type == Sensor.TYPE_ACCELEROMETER) {
            latestAx = event.values[0]
            latestAy = event.values[1]
            latestAz = event.values[2]

            val reading = RawSensorReading(
                timestamp = nowSeconds,
                ax = latestAx,
                ay = latestAy,
                az = latestAz,
                gx = latestGx,
                gy = latestGy,
                gz = latestGz
            )

            // Maintain bounded 5-second ring buffer (sample 251 drops sample 1)
            rollingBuffer.addLast(reading)
            while (rollingBuffer.size > maxBufferSamples) {
                rollingBuffer.pollFirst()
            }

            onReadingListener?.invoke(reading)

        } else if (event.sensor.type == Sensor.TYPE_GYROSCOPE) {
            latestGx = event.values[0]
            latestGy = event.values[1]
            latestGz = event.values[2]
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {
        // No-op
    }
}
