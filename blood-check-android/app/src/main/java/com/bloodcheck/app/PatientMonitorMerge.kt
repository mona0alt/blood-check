package com.bloodcheck.app

/**
 * 将本地预测结果合并为监测页状态，字段优先级对齐 uniapp [fetchMonitorData]。
 */
data class PatientMonitorModel(
    val patientId: String,
    val patientInfo: PatientInfo?,
    val hemoglobin: MetricResult?,
    val lactate: MetricResult?,
    val perfusionIndex: PerfusionIndexResult?
)

object PatientMonitorMerge {

    fun merge(
        prev: PatientMonitorModel?,
        response: PredictionResponse,
        inputPatientId: String
    ): PatientMonitorModel {
        val pred = requireNotNull(response.prediction) { "prediction" }

        val mergedInfo: PatientInfo? = when {
            response.patientInfo != null ->
                prev?.patientInfo?.mergedWith(response.patientInfo) ?: response.patientInfo
            else -> prev?.patientInfo
        }

        val hb = pred.hemoglobin
            ?: metricFromMap(mergedInfo, "血红蛋白", "g/L")
            ?: prev?.hemoglobin

        val lac = pred.lactate
            ?: metricFromMap(mergedInfo, "乳酸", "mmol/L")
            ?: prev?.lactate

        val perf = pred.perfusionIndex
            ?: perfusionFromMap(mergedInfo, "血流灌注指数")
            ?: prev?.perfusionIndex

        val pid = pred.patientId
            ?: mergedInfo?.get("统一ID")
            ?: mergedInfo?.get("编号")
            ?: inputPatientId.trim().ifBlank { prev?.patientId ?: "" }

        return PatientMonitorModel(
            patientId = pid,
            patientInfo = mergedInfo,
            hemoglobin = hb,
            lactate = lac,
            perfusionIndex = perf
        )
    }

    private fun metricFromMap(info: PatientInfo?, key: String, unit: String): MetricResult? {
        val raw = info?.get(key) ?: return null
        val v = raw.toDoubleOrNull() ?: return null
        return MetricResult(v, unit, null)
    }

    private fun perfusionFromMap(info: PatientInfo?, key: String): PerfusionIndexResult? {
        val raw = info?.get(key) ?: return null
        val v = raw.toDoubleOrNull() ?: return null
        return PerfusionIndexResult(v, null, null)
    }
}
