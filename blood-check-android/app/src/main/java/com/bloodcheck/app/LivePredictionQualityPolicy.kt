package com.bloodcheck.app

data class LivePredictionQualityDecision(
    val accepted: Boolean,
    val message: String? = null
)

object LivePredictionQualityPolicy {
    private const val LOW_SPO2_THRESHOLD = 90.0
    private const val MIN_RELIABLE_LOW_SPO2_QUALITY = 0.85

    fun validate(response: PredictionResponse): LivePredictionQualityDecision {
        val info = response.patientInfo ?: return LivePredictionQualityDecision(accepted = true)
        val spo2 = info["血氧饱和度"]?.toDoubleOrNull()
        val quality = info["信号质量"]?.toDoubleOrNull()

        if (spo2 != null && spo2 < LOW_SPO2_THRESHOLD) {
            val reliable = quality != null && quality >= MIN_RELIABLE_LOW_SPO2_QUALITY
            if (!reliable) {
                return LivePredictionQualityDecision(
                    accepted = false,
                    message = "信号质量不足，血氧结果不可靠，请调整探头后重新采集"
                )
            }
        }

        return LivePredictionQualityDecision(accepted = true)
    }
}
