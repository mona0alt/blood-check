package com.bloodcheck.app

class LiveMonitorGate(
    private val minRows: Int,
    private val stableWindowMillis: Long = 0L
) {
    var isOpen: Boolean = false
        private set

    init {
        require(minRows > 0) { "minRows must be positive" }
        require(stableWindowMillis >= 0L) { "stableWindowMillis must not be negative" }
    }

    fun reset() {
        isOpen = false
    }

    fun markSamples(sampleCount: Int, stableElapsedMillis: Long = Long.MAX_VALUE): Boolean {
        if (isOpen || sampleCount < minRows || stableElapsedMillis < stableWindowMillis) return false
        isOpen = true
        return true
    }
}
