package com.bloodcheck.app

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class StartupMonitorRestorePolicyTest {
    @Test
    fun restoresPatientIdentityButLeavesHistoricalMetricsEmpty() {
        val storedInfo = PatientInfo(
            mapOf(
                "姓名" to "测试",
                "性别" to "男",
                "年龄" to "33",
                "住院号" to "11",
                "心率" to "55",
                "血氧饱和度" to "84"
            )
        )

        val model = StartupMonitorRestorePolicy.restoreHomeModel(
            currentPatientId = "11",
            latestPatientId = "11",
            storedPatientInfo = storedInfo,
            localPatch = PatientInfo(emptyMap())
        )

        requireNotNull(model)
        assertEquals("11", model.patientId)
        assertEquals("测试", model.patientInfo?.get("姓名"))
        assertEquals("33", model.patientInfo?.get("年龄"))
        assertNull(model.patientInfo?.get("心率"))
        assertNull(model.patientInfo?.get("血氧饱和度"))
        assertNull(model.hemoglobin)
        assertNull(model.lactate)
        assertNull(model.perfusionIndex)
    }

    @Test
    fun localProfileOverridesStoredIdentityFields() {
        val storedInfo = PatientInfo(
            mapOf(
                "姓名" to "旧姓名",
                "性别" to "男",
                "年龄" to "33",
                "住院号" to "11"
            )
        )

        val model = StartupMonitorRestorePolicy.restoreHomeModel(
            currentPatientId = "11",
            latestPatientId = "11",
            storedPatientInfo = storedInfo,
            localPatch = PatientInfo(mapOf("姓名" to "新姓名", "年龄" to "40", "住院号" to "11"))
        )

        requireNotNull(model)
        assertEquals("新姓名", model.patientInfo?.get("姓名"))
        assertEquals("40", model.patientInfo?.get("年龄"))
        assertEquals("男", model.patientInfo?.get("性别"))
    }
}
