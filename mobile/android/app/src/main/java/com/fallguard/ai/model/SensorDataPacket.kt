package com.fallguard.ai.model

import com.google.gson.annotations.SerializedName

/**
 * Clean versioned sensor data packet sent over WebSocket at 50 Hz.
 * Units:
 * accelerometer: m/s^2 (Android native output)
 * gyroscope: rad/s (Android native output)
 */
data class Axis3D(
    @SerializedName("x") val x: Float,
    @SerializedName("y") val y: Float,
    @SerializedName("z") val z: Float
)

data class SensorDataPacket(
    @SerializedName("type") val type: String = "sensor_data",
    @SerializedName("version") val version: Int = 1,
    @SerializedName("device_id") val deviceId: String,
    @SerializedName("timestamp") val timestamp: Double,
    @SerializedName("unit") val unit: String = "m/s2",
    @SerializedName("accelerometer") val accelerometer: Axis3D,
    @SerializedName("gyroscope") val gyroscope: Axis3D,
    @SerializedName("location") val location: Map<String, Double>? = null
)

data class RawSensorReading(
    val timestamp: Double,
    val ax: Float,
    val ay: Float,
    val az: Float,
    val gx: Float,
    val gy: Float,
    val gz: Float
)
