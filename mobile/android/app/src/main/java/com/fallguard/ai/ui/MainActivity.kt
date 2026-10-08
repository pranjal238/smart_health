package com.fallguard.ai.ui

import android.Manifest
import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.content.ServiceConnection
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.IBinder
import android.os.Looper
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.fallguard.ai.R
import com.fallguard.ai.service.SensorForegroundService
import com.google.android.material.button.MaterialButton
import com.google.android.material.textfield.TextInputEditText

/**
 * FallGuard AI - Smartphone Physical Sensor Client Dashboard
 */
class MainActivity : AppCompatActivity() {

    private val PERMISSIONS_REQUEST_CODE = 200

    private lateinit var tvStatus: TextView
    private lateinit var tvBufferInfo: TextView
    private lateinit var tvSensorPreview: TextView
    private lateinit var etServerUrl: TextInputEditText
    private lateinit var etDeviceId: TextInputEditText
    private lateinit var etJwtToken: TextInputEditText
    private lateinit var btnToggleMonitoring: MaterialButton
    private lateinit var btnSimulateFall: MaterialButton

    private var service: SensorForegroundService? = null
    private var isMonitoringActive = false
    private val mainHandler = Handler(Looper.getMainLooper())

    private val connection = object : ServiceConnection {
        override fun onServiceConnected(name: ComponentName?, binder: IBinder?) {
            service = (binder as? SensorForegroundService.LocalBinder)?.getService()
            setupServiceListeners()
        }

        override fun onServiceDisconnected(name: ComponentName?) {
            service = null
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        initViews()
        requestAppPermissions()
        startStatsUpdateLoop()
    }

    private fun initViews() {
        tvStatus = findViewById(R.id.tv_connection_status)
        tvBufferInfo = findViewById(R.id.tv_buffer_info)
        tvSensorPreview = findViewById(R.id.tv_sensor_preview)
        etServerUrl = findViewById(R.id.et_server_url)
        etDeviceId = findViewById(R.id.et_device_id)
        etJwtToken = findViewById(R.id.et_jwt_token)
        btnToggleMonitoring = findViewById(R.id.btn_toggle_monitoring)
        btnSimulateFall = findViewById(R.id.btn_simulate_fall)

        // Pre-fill demo token placeholder
        etJwtToken.setText("demo_token")

        btnToggleMonitoring.setOnClickListener {
            if (isMonitoringActive) {
                stopMonitoring()
            } else {
                startMonitoring()
            }
        }

        btnSimulateFall.setOnClickListener {
            val intent = Intent(this, FallConfirmationActivity::class.java).apply {
                putExtra("EVENT_ID", 9999)
                putExtra("TIMEOUT_SECONDS", 15f)
                putExtra("CONFIDENCE", 0.95f)
            }
            startActivity(intent)
        }
    }

    private fun startMonitoring() {
        val serverUrl = etServerUrl.text.toString().trim()
        val deviceId = etDeviceId.text.toString().trim()
        val jwtToken = etJwtToken.text.toString().trim()

        if (serverUrl.isEmpty() || deviceId.isEmpty()) {
            Toast.makeText(this, "Please enter Server URL and Device ID", Toast.LENGTH_SHORT).show()
            return
        }

        val serviceIntent = Intent(this, SensorForegroundService::class.java).apply {
            putExtra("SERVER_URL", serverUrl)
            putExtra("DEVICE_ID", deviceId)
            putExtra("JWT_TOKEN", jwtToken)
        }

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            startForegroundService(serviceIntent)
        } else {
            startService(serviceIntent)
        }

        bindService(serviceIntent, connection, Context.BIND_AUTO_CREATE)
        isMonitoringActive = true
        btnToggleMonitoring.text = getString(R.string.btn_stop)
        btnToggleMonitoring.setBackgroundColor(ContextCompat.getColor(this, R.color.danger))
    }

    private fun stopMonitoring() {
        val serviceIntent = Intent(this, SensorForegroundService::class.java)
        try {
            unbindService(connection)
        } catch (e: Exception) {
            // Ignore
        }
        stopService(serviceIntent)
        isMonitoringActive = false
        btnToggleMonitoring.text = getString(R.string.btn_start)
        btnToggleMonitoring.setBackgroundColor(ContextCompat.getColor(this, R.color.primary))
        tvStatus.text = getString(R.string.status_disconnected)
        tvStatus.setTextColor(ContextCompat.getColor(this, R.color.danger))
    }

    private fun setupServiceListeners() {
        service?.webSocketClient?.onConnectionStateChanged = { connected ->
            if (connected) {
                tvStatus.text = getString(R.string.status_connected)
                tvStatus.setTextColor(ContextCompat.getColor(this, R.color.success))
            } else {
                tvStatus.text = getString(R.string.status_disconnected)
                tvStatus.setTextColor(ContextCompat.getColor(this, R.color.danger))
            }
        }
    }

    private fun startStatsUpdateLoop() {
        mainHandler.post(object : Runnable {
            override fun run() {
                service?.let { s ->
                    val sc = s.sensorCollector
                    val ax = sc.latestAx
                    val ay = sc.latestAy
                    val az = sc.latestAz
                    val gx = sc.latestGx
                    val gy = sc.latestGy
                    val gz = sc.latestGz
                    val bufCount = sc.rollingBuffer.size

                    tvSensorPreview.text = String.format(
                        "Acc:  (%.2f, %.2f, %.2f) m/s²\nGyro: (%.2f, %.2f, %.2f) rad/s",
                        ax, ay, az, gx, gy, gz
                    )
                    tvBufferInfo.text = String.format(
                        "Rolling Buffer: %d / 250 samples (~%.1fs)",
                        bufCount, bufCount / 50.0
                    )
                }
                mainHandler.postDelayed(this, 200) // Update UI at 5 Hz
            }
        })
    }

    private fun requestAppPermissions() {
        val needed = mutableListOf<String>()

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.POST_NOTIFICATIONS) != PackageManager.PERMISSION_GRANTED) {
                needed.add(Manifest.permission.POST_NOTIFICATIONS)
            }
        }
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.BODY_SENSORS) != PackageManager.PERMISSION_GRANTED) {
            needed.add(Manifest.permission.BODY_SENSORS)
        }
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.ACCESS_FINE_LOCATION) != PackageManager.PERMISSION_GRANTED) {
            needed.add(Manifest.permission.ACCESS_FINE_LOCATION)
        }

        if (needed.isNotEmpty()) {
            ActivityCompat.requestPermissions(this, needed.toTypedArray(), PERMISSIONS_REQUEST_CODE)
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        if (isMonitoringActive) {
            try {
                unbindService(connection)
            } catch (e: Exception) {
                // Ignore
            }
        }
    }
}
