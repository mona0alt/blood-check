package com.bloodcheck.app

import org.junit.Assert.assertEquals
import org.junit.Test

class BleMonitoringStartPolicyTest {
    @Test
    fun reusesConnectedStatusSessionWhenStartingCollection() {
        val action = BleMonitoringStartPolicy.choose(statusSessionReadyToCollect = true)

        assertEquals(BleMonitoringStartAction.REUSE_CONNECTED_STATUS_SESSION, action)
    }

    @Test
    fun startsNewScanWhenStatusSessionIsNotReady() {
        val action = BleMonitoringStartPolicy.choose(statusSessionReadyToCollect = false)

        assertEquals(BleMonitoringStartAction.START_NEW_COLLECTING_SCAN, action)
    }
}
