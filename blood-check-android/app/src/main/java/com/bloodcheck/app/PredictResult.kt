package com.bloodcheck.app

import org.json.JSONObject

data class PredictionResponse(
    val success: Boolean,
    val patientInfo: PatientInfo?,
    val prediction: Prediction?,
    val error: String?
) {
    companion object {
        fun fromJson(json: String): PredictionResponse {
            val root = JSONObject(json)
            return PredictionResponse(
                success = root.optBoolean("success"),
                patientInfo = root.optJSONObject("patient_info")?.toPatientInfo(),
                prediction = root.optJSONObject("prediction")?.toPrediction(),
                error = root.optString("error").takeIf { it.isNotBlank() && it != "null" }
            )
        }
    }
}

/**
 * 完整保留 [patient_info] 下所有键值（字符串化），与 uniapp 展示/导出字段对齐。
 */
data class PatientInfo(
    val values: Map<String, String>
) {
    operator fun get(key: String): String? = values[key]

    val id: String? get() = values["编号"] ?: values["统一ID"]
    val name: String? get() = values["姓名"]
    val gender: String? get() = values["性别"]
    val age: String? get() = values["年龄"]
    val heartRate: String? get() = values["心率"]
    val bloodOxygen: String? get() = values["血氧饱和度"]
    val temperature: String? get() = values["体温"]

    fun mergedWith(incoming: PatientInfo?): PatientInfo {
        if (incoming == null) return this
        val m = values.toMutableMap()
        m.putAll(incoming.values)
        return PatientInfo(m)
    }

    fun toJsonObject(): JSONObject {
        val o = JSONObject()
        values.forEach { (k, v) -> o.put(k, v) }
        return o
    }

    companion object {
        fun fromJsonObject(obj: JSONObject?): PatientInfo? {
            if (obj == null || obj.length() == 0) return null
            val map = mutableMapOf<String, String>()
            val it = obj.keys()
            while (it.hasNext()) {
                val key = it.next()
                obj.optAnyAsString(key)?.takeIf { it.isNotBlank() }?.let { map[key] = it }
            }
            return if (map.isEmpty()) null else PatientInfo(map)
        }
    }
}

data class Prediction(
    val patientId: String?,
    val hemoglobin: MetricResult?,
    val perfusionIndex: PerfusionIndexResult?,
    val lactate: MetricResult?,
    val predictionTime: String?,
    val predictionId: String?
)

data class MetricResult(
    val value: Double?,
    val unit: String?,
    val clinicalInterpretation: String?
)

data class PerfusionIndexResult(
    val value: Double?,
    val classification: String?,
    val interpretation: String?
)

private fun JSONObject.toPatientInfo(): PatientInfo? = PatientInfo.fromJsonObject(this)

private fun JSONObject.toPrediction(): Prediction {
    return Prediction(
        patientId = optString("patient_id").takeIf { it.isNotBlank() },
        hemoglobin = optJSONObject("hemoglobin")?.toMetricResult(),
        perfusionIndex = optJSONObject("perfusion_index")?.toPerfusionIndexResult(),
        lactate = optJSONObject("lactate")?.toMetricResult(),
        predictionTime = optString("prediction_time").takeIf { it.isNotBlank() },
        predictionId = optString("prediction_id").takeIf { it.isNotBlank() }
    )
}

private fun JSONObject.toMetricResult(): MetricResult {
    return MetricResult(
        value = optDoubleOrNull("value"),
        unit = optString("unit").takeIf { it.isNotBlank() },
        clinicalInterpretation = optString("clinical_interpretation").takeIf { it.isNotBlank() }
    )
}

private fun JSONObject.toPerfusionIndexResult(): PerfusionIndexResult {
    return PerfusionIndexResult(
        value = optDoubleOrNull("value"),
        classification = optString("classification").takeIf { it.isNotBlank() },
        interpretation = optString("interpretation").takeIf { it.isNotBlank() }
    )
}

private fun JSONObject.optAnyAsString(key: String): String? {
    if (isNull(key) || !has(key)) return null
    return opt(key)?.toString()?.takeIf { it.isNotBlank() }
}

private fun JSONObject.optDoubleOrNull(key: String): Double? {
    if (isNull(key) || !has(key)) return null
    return optDouble(key).takeIf { !it.isNaN() }
}

fun MetricResult.toJsonObject(): JSONObject = JSONObject().apply {
    put("value", value)
    put("unit", unit)
    put("clinical_interpretation", clinicalInterpretation)
}

fun PerfusionIndexResult.toJsonObject(): JSONObject = JSONObject().apply {
    put("value", value)
    put("classification", classification)
    put("interpretation", interpretation)
}

fun Prediction.toJsonObject(): JSONObject = JSONObject().apply {
    put("patient_id", patientId)
    put("hemoglobin", hemoglobin?.toJsonObject())
    put("perfusion_index", perfusionIndex?.toJsonObject())
    put("lactate", lactate?.toJsonObject())
    put("prediction_time", predictionTime)
    put("prediction_id", predictionId)
}
