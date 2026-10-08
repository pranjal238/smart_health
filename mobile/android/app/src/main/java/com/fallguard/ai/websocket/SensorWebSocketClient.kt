package com.fallguard.ai.websocket

import android.os.Handler
import android.os.Looper
import android.util.Log
import com.fallguard.ai.model.Axis3D
import com.fallguard.ai.model.FallConfirmationResponse
import com.fallguard.ai.model.IncomingPendingFallMessage
import com.fallguard.ai.model.RawSensorReading
import com.fallguard.ai.model.SensorDataPacket
import com.google.gson.Gson
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.Response
import okhttp3.WebSocket
import okhttp3.WebSocketListener
import java.util.concurrent.TimeUnit

/**
 * Persistent live WebSocket client for streaming 50Hz smartphone telemetry.
 * Automatically recovers with exponential backoff reconnection upon network drops.
 */
class SensorWebSocketClient(
    private var serverUrl: String,
    private var jwtToken: String,
    private var deviceId: String
) {
    private val TAG = "SensorWSClient"
    private val gson = Gson()
    private val client = OkHttpClient.Builder()
        .readTimeout(0, TimeUnit.MILLISECONDS)
        .pingInterval(15, TimeUnit.SECONDS)
        .build()

    private var webSocket: WebSocket? = null
    private val mainHandler = Handler(Looper.getMainLooper())

    @Volatile var isConnected = false
        private set

    private var shouldReconnect = true
    private var reconnectDelayMs = 1000L
    private val MAX_RECONNECT_DELAY_MS = 16000L

    var onConnectionStateChanged: ((Boolean) -> Unit)? = null
    var onPendingFallReceived: ((IncomingPendingFallMessage) -> Unit)? = null

    fun updateConfig(url: String, token: String, devId: String) {
        serverUrl = url
        jwtToken = token
        deviceId = devId
    }

    fun connect() {
        shouldReconnect = true
        disconnectInternal()

        val fullUrl = if (serverUrl.contains("?")) {
            "$serverUrl&token=$jwtToken&device_id=$deviceId"
        } else {
            "$serverUrl?token=$jwtToken&device_id=$deviceId"
        }

        val request = Request.Builder()
            .url(fullUrl)
            .build()

        webSocket = client.newWebSocket(request, object : WebSocketListener() {
            override fun onOpen(ws: WebSocket, response: Response) {
                Log.i(TAG, "WebSocket connected to $serverUrl")
                isConnected = true
                reconnectDelayMs = 1000L
                mainHandler.post { onConnectionStateChanged?.invoke(true) }
            }

            override fun onMessage(ws: WebSocket, text: String) {
                try {
                    val msg = gson.fromJson(text, IncomingPendingFallMessage::class.java)
                    if (msg.type == "FALL_PENDING_CONFIRMATION" && msg.event != null) {
                        Log.w(TAG, "Received potential fall alert from backend! Event ID: ${msg.event.id}")
                        mainHandler.post { onPendingFallReceived?.invoke(msg) }
                    }
                } catch (e: Exception) {
                    Log.d(TAG, "Incoming WS message: $text")
                }
            }

            override fun onClosing(ws: WebSocket, code: Int, reason: String) {
                Log.w(TAG, "WebSocket closing ($code): $reason")
                isConnected = false
                mainHandler.post { onConnectionStateChanged?.invoke(false) }
            }

            override fun onClosed(ws: WebSocket, code: Int, reason: String) {
                Log.w(TAG, "WebSocket closed ($code): $reason")
                isConnected = false
                mainHandler.post { onConnectionStateChanged?.invoke(false) }
                scheduleReconnect()
            }

            override fun onFailure(ws: WebSocket, t: Throwable, response: Response?) {
                Log.e(TAG, "WebSocket error: ${t.message}")
                isConnected = false
                mainHandler.post { onConnectionStateChanged?.invoke(false) }
                scheduleReconnect()
            }
        })
    }

    fun disconnect() {
        shouldReconnect = false
        disconnectInternal()
    }

    private fun disconnectInternal() {
        try {
            webSocket?.close(1000, "Normal closure")
            webSocket = null
        } catch (e: Exception) {
            // Ignore
        }
        isConnected = false
    }

    private fun scheduleReconnect() {
        if (!shouldReconnect) return
        mainHandler.postDelayed({
            if (shouldReconnect && !isConnected) {
                Log.i(TAG, "Attempting automatic reconnection (delay ${reconnectDelayMs}ms)...")
                reconnectDelayMs = (reconnectDelayMs * 2).coerceAtMost(MAX_RECONNECT_DELAY_MS)
                connect()
            }
        }, reconnectDelayMs)
    }

    fun sendSensorReading(reading: RawSensorReading, location: Map<String, Double>? = null): Boolean {
        if (!isConnected) return false
        val packet = SensorDataPacket(
            deviceId = deviceId,
            timestamp = reading.timestamp,
            unit = "m/s2",
            accelerometer = Axis3D(reading.ax, reading.ay, reading.az),
            gyroscope = Axis3D(reading.gx, reading.gy, reading.gz),
            location = location
        )
        val json = gson.toJson(packet)
        return webSocket?.send(json) ?: false
    }

    fun sendFallConfirmation(eventId: Int, action: String, location: Map<String, Double>? = null): Boolean {
        val payload = FallConfirmationResponse(
            eventId = eventId,
            action = action,
            location = location
        )
        val json = gson.toJson(payload)
        Log.i(TAG, "Sending fall confirmation response: $json")
        return webSocket?.send(json) ?: false
    }
}
