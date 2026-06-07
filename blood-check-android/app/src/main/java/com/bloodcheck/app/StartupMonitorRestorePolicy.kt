package com.bloodcheck.app

import org.json.JSONObject

object StartupMonitorRestorePolicy {
    private val STARTUP_PATIENT_INFO_KEYS = setOf(
        "编号",
        "统一ID",
        "住院号",
        "姓名",
        "性别",
        "年龄",
        "体温",
        "房颤"
    )

    fun restoreHomeModel(
        currentPatientId: String,
        latestRecord: JSONObject?,
        localPatch: PatientInfo
    ): PatientMonitorModel? {
        val storedInfo = latestRecord?.optJSONObject("patient_info")
            ?.let { PatientInfo.fromJsonObject(it) }
        return restoreHomeModel(
            currentPatientId = currentPatientId,
            latestPatientId = latestRecord?.optString("patient_id").orEmpty(),
            storedPatientInfo = storedInfo,
            localPatch = localPatch
        )
    }

    fun restoreHomeModel(
        currentPatientId: String,
        latestPatientId: String,
        storedPatientInfo: PatientInfo?,
        localPatch: PatientInfo
    ): PatientMonitorModel? {
        val mergedInfo = storedPatientInfo?.identityOnly()?.mergedWith(localPatch)
            ?: localPatch.takeIf { it.values.isNotEmpty() }
        val patientId = currentPatientId.trim()
            .ifBlank { latestPatientId.trim() }
            .ifBlank { mergedInfo?.get("住院号").orEmpty() }
            .ifBlank { mergedInfo?.get("统一ID").orEmpty() }
            .ifBlank { mergedInfo?.get("编号").orEmpty() }

        if (patientId.isBlank() && mergedInfo == null) return null
        return PatientMonitorModel(
            patientId = patientId,
            patientInfo = mergedInfo,
            hemoglobin = null,
            lactate = null,
            perfusionIndex = null
        )
    }

    private fun PatientInfo.identityOnly(): PatientInfo? {
        val filtered = values.filterKeys { it in STARTUP_PATIENT_INFO_KEYS }
        return filtered.takeIf { it.isNotEmpty() }?.let { PatientInfo(it) }
    }
}
