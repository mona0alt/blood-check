package com.bloodcheck.app

import android.Manifest
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class BlePermissionPolicyTest {
    @Test
    fun androidTwelveAndAboveRequiresNearbyDevicePermissions() {
        val permissions = BlePermissionPolicy.requiredPermissions(31).toList()

        assertEquals(
            listOf(
                Manifest.permission.BLUETOOTH_SCAN,
                Manifest.permission.BLUETOOTH_CONNECT
            ),
            permissions
        )
        assertFalse(permissions.contains(Manifest.permission.ACCESS_FINE_LOCATION))
    }

    @Test
    fun androidElevenAndBelowRequiresLocationPermission() {
        val permissions = BlePermissionPolicy.requiredPermissions(30).toList()

        assertEquals(listOf(Manifest.permission.ACCESS_FINE_LOCATION), permissions)
    }

    @Test
    fun reportsMissingPermissionsForCurrentAndroidVersion() {
        val missing = BlePermissionPolicy.missingPermissions(31) { permission ->
            permission == Manifest.permission.BLUETOOTH_CONNECT
        }

        assertEquals(listOf(Manifest.permission.BLUETOOTH_SCAN), missing.toList())
    }

    @Test
    fun androidTwelveDeniedMessageMentionsNearbyDevices() {
        assertTrue(BlePermissionPolicy.deniedStatusMessage(31).contains("附近设备"))
    }

    @Test
    fun androidElevenDeniedMessageMentionsLocation() {
        assertTrue(BlePermissionPolicy.deniedStatusMessage(30).contains("定位"))
    }
}
