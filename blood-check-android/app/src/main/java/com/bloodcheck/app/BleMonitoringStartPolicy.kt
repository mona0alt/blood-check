package com.bloodcheck.app

enum class BleMonitoringStartAction {
    REUSE_CONNECTED_STATUS_SESSION,
    START_NEW_COLLECTING_SCAN
}

object BleMonitoringStartPolicy {
    fun choose(statusSessionReadyToCollect: Boolean): BleMonitoringStartAction {
        return if (statusSessionReadyToCollect) {
            BleMonitoringStartAction.REUSE_CONNECTED_STATUS_SESSION
        } else {
            BleMonitoringStartAction.START_NEW_COLLECTING_SCAN
        }
    }
}
