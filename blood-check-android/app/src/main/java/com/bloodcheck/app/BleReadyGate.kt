package com.bloodcheck.app

enum class BleReadiness {
    READY,
    NEED_PERMISSIONS,
    NEED_LOCATION_SERVICE
}

object BleReadyGate {
    fun evaluate(
        missingPermissions: Array<String>,
        locationServiceReady: Boolean
    ): BleReadiness {
        return when {
            missingPermissions.isNotEmpty() -> BleReadiness.NEED_PERMISSIONS
            !locationServiceReady -> BleReadiness.NEED_LOCATION_SERVICE
            else -> BleReadiness.READY
        }
    }
}
