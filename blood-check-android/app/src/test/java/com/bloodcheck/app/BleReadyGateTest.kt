package com.bloodcheck.app

import org.junit.Assert.assertEquals
import org.junit.Test

class BleReadyGateTest {
    @Test
    fun readyWhenPermissionsAndLocationAreReady() {
        val state = BleReadyGate.evaluate(
            missingPermissions = emptyArray(),
            locationServiceReady = true
        )

        assertEquals(BleReadiness.READY, state)
    }

    @Test
    fun missingPermissionTakesPriority() {
        val state = BleReadyGate.evaluate(
            missingPermissions = arrayOf("android.permission.BLUETOOTH_SCAN"),
            locationServiceReady = false
        )

        assertEquals(BleReadiness.NEED_PERMISSIONS, state)
    }

    @Test
    fun needsLocationWhenPermissionsAreReadyButLocationIsOff() {
        val state = BleReadyGate.evaluate(
            missingPermissions = emptyArray(),
            locationServiceReady = false
        )

        assertEquals(BleReadiness.NEED_LOCATION_SERVICE, state)
    }
}
