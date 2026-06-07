package com.bloodcheck.app

import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class LivePredictionQualityPolicyTest {
    @Test
    fun rejectsLowSpo2WhenSignalQualityIsNotReliable() {
        val response = predictionResponse(
            "血氧饱和度" to "85",
            "信号质量" to "0.8"
        )

        val decision = LivePredictionQualityPolicy.validate(response)

        assertFalse(decision.accepted)
        assertTrue(decision.message!!.contains("信号质量不足"))
    }

    @Test
    fun acceptsLowSpo2WhenSignalQualityIsReliable() {
        val response = predictionResponse(
            "血氧饱和度" to "85",
            "信号质量" to "0.91"
        )

        val decision = LivePredictionQualityPolicy.validate(response)

        assertTrue(decision.accepted)
        assertNull(decision.message)
    }

    @Test
    fun acceptsNormalSpo2EvenWhenSignalQualityIsOnlyModerate() {
        val response = predictionResponse(
            "血氧饱和度" to "96",
            "信号质量" to "0.8"
        )

        val decision = LivePredictionQualityPolicy.validate(response)

        assertTrue(decision.accepted)
        assertNull(decision.message)
    }

    private fun predictionResponse(vararg values: Pair<String, String>): PredictionResponse {
        return PredictionResponse(
            success = true,
            patientInfo = PatientInfo(mapOf(*values)),
            prediction = null,
            error = null
        )
    }
}
