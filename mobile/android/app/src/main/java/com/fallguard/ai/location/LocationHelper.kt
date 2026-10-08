package com.fallguard.ai.location

import android.annotation.SuppressLint
import android.content.Context
import android.content.pm.PackageManager
import android.location.Location
import androidx.core.content.ContextCompat
import com.google.android.gms.location.LocationServices
import com.google.android.gms.location.Priority
import com.google.android.gms.tasks.CancellationTokenSource

/**
 * Privacy-preserving location helper.
 * Strictly acquires location on-demand ONLY during a confirmed fall incident.
 * Zero continuous location logging or background GPS tracking.
 */
class LocationHelper(private val context: Context) {

    private val fusedClient = LocationServices.getFusedLocationProviderClient(context)

    fun hasLocationPermission(): Boolean {
        return ContextCompat.checkSelfPermission(
            context,
            android.Manifest.permission.ACCESS_FINE_LOCATION
        ) == PackageManager.PERMISSION_GRANTED
    }

    @SuppressLint("MissingPermission")
    fun getEmergencyLocation(callback: (Map<String, Double>?) -> Unit) {
        if (!hasLocationPermission()) {
            callback(null)
            return
        }

        try {
            val cts = CancellationTokenSource()
            fusedClient.getCurrentLocation(Priority.PRIORITY_HIGH_ACCURACY, cts.token)
                .addOnSuccessListener { loc: Location? ->
                    if (loc != null) {
                        callback(mapOf("lat" to loc.latitude, "lon" to loc.longitude))
                    } else {
                        // Fallback to last known
                        fusedClient.lastLocation.addOnSuccessListener { last: Location? ->
                            if (last != null) {
                                callback(mapOf("lat" to last.latitude, "lon" to last.longitude))
                            } else {
                                callback(null)
                            }
                        }.addOnFailureListener { callback(null) }
                    }
                }
                .addOnFailureListener { callback(null) }
        } catch (e: Exception) {
            callback(null)
        }
    }
}
