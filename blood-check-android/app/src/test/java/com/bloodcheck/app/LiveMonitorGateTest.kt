package com.bloodcheck.app

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class LiveMonitorGateTest {
    @Test
    fun waitsUntilMinimumRowsBeforeOpeningInitialPoll() {
        val gate = LiveMonitorGate(minRows = 200)

        assertFalse(gate.markSamples(199))
        assertTrue(gate.markSamples(200))
    }

    @Test
    fun opensInitialPollOnlyOncePerSession() {
        val gate = LiveMonitorGate(minRows = 200)

        assertTrue(gate.markSamples(200))
        assertFalse(gate.markSamples(250))
    }

    @Test
    fun resetAllowsNextMonitoringSessionToOpenAgain() {
        val gate = LiveMonitorGate(minRows = 200)

        assertTrue(gate.markSamples(200))
        gate.reset()

        assertFalse(gate.markSamples(100))
        assertTrue(gate.markSamples(200))
    }

    @Test
    fun waitsForStableWindowBeforeOpeningInitialPoll() {
        val gate = LiveMonitorGate(minRows = 200, stableWindowMillis = 10_000L)

        assertFalse(gate.markSamples(sampleCount = 250, stableElapsedMillis = 9_999L))
        assertTrue(gate.markSamples(sampleCount = 250, stableElapsedMillis = 10_000L))
    }

    @Test
    fun resetRequiresStableWindowAgain() {
        val gate = LiveMonitorGate(minRows = 200, stableWindowMillis = 10_000L)

        assertTrue(gate.markSamples(sampleCount = 250, stableElapsedMillis = 10_000L))
        gate.reset()

        assertFalse(gate.markSamples(sampleCount = 250, stableElapsedMillis = 5_000L))
        assertTrue(gate.markSamples(sampleCount = 250, stableElapsedMillis = 10_000L))
    }
}
