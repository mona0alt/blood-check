package com.bloodcheck.app

import android.content.Context
import org.json.JSONArray
import org.json.JSONObject
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * 对齐 uniapp [dataStorage.js]：键名 `patient_data_{patientId}`，值为记录数组。
 */
data class LocalCollectProfile(
    val name: String = "",
    val gender: String = "",
    val age: String = "",
    val hospitalId: String = "",
    val temperature: String = "",
    val atrialFibrillationYes: Boolean = false,
) {
    fun toPatientInfoPatch(): PatientInfo {
        val m = LinkedHashMap<String, String>()
        if (name.isNotBlank()) m["姓名"] = name.trim()
        if (gender.isNotBlank()) m["性别"] = gender.trim()
        if (age.isNotBlank()) m["年龄"] = age.trim()
        if (hospitalId.isNotBlank()) m["住院号"] = hospitalId.trim()
        if (temperature.isNotBlank()) m["体温"] = temperature.trim()
        if (m.isNotEmpty() || atrialFibrillationYes) {
            m["房颤"] = if (atrialFibrillationYes) "是" else "否"
        }
        return PatientInfo(m)
    }

    fun toJson(): JSONObject {
        val o = JSONObject()
        o.put("name", name)
        o.put("gender", gender)
        o.put("age", age)
        o.put("hospitalId", hospitalId)
        o.put("temperature", temperature)
        o.put("afYes", atrialFibrillationYes)
        return o
    }

    companion object {
        fun fromJson(o: JSONObject?): LocalCollectProfile {
            if (o == null) return LocalCollectProfile()
            return LocalCollectProfile(
                name = o.optString("name"),
                gender = o.optString("gender"),
                age = o.optString("age"),
                hospitalId = o.optString("hospitalId"),
                temperature = o.optString("temperature"),
                atrialFibrillationYes = o.optBoolean("afYes", false),
            )
        }
    }
}

object PatientDataStorage {

    private const val PREFS_NAME = "hemoglobin_monitor_prefs"
    private const val KEY_COLLECTION_PERIOD = "collectionPeriod"
    private const val KEY_HEMOGLOBIN_THRESHOLD = "hemoglobinThreshold"
    private const val KEY_PATIENT_ID = "patientId"
    private const val KEY_LOCAL_COLLECT_PROFILE = "localCollectProfileJson"
    private const val STORAGE_KEY_PREFIX = "patient_data_"
    private const val MAX_RECORDS = 10_000

    private fun prefs(ctx: Context) =
        ctx.applicationContext.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)

    fun loadCollectionPeriodSeconds(ctx: Context): Int {
        val v = prefs(ctx).getInt(KEY_COLLECTION_PERIOD, 0)
        return if (v > 0) v else 5
    }

    fun loadHemoglobinThreshold(ctx: Context): Int {
        val v = prefs(ctx).getInt(KEY_HEMOGLOBIN_THRESHOLD, 0)
        return if (v > 0) v else 70
    }

    fun saveSettings(ctx: Context, collectionPeriodSeconds: Int, hemoglobinThreshold: Int) {
        prefs(ctx).edit()
            .putInt(KEY_COLLECTION_PERIOD, collectionPeriodSeconds)
            .putInt(KEY_HEMOGLOBIN_THRESHOLD, hemoglobinThreshold)
            .apply()
    }

    fun loadPatientId(ctx: Context): String? =
        prefs(ctx).getString(KEY_PATIENT_ID, null)?.takeIf { it.isNotBlank() }

    fun savePatientId(ctx: Context, patientId: String) {
        prefs(ctx).edit().putString(KEY_PATIENT_ID, patientId.trim()).apply()
    }

    fun loadLocalCollectProfile(ctx: Context): LocalCollectProfile {
        val raw = prefs(ctx).getString(KEY_LOCAL_COLLECT_PROFILE, null) ?: return LocalCollectProfile()
        return try {
            LocalCollectProfile.fromJson(JSONObject(raw))
        } catch (_: Exception) {
            LocalCollectProfile()
        }
    }

    fun saveLocalCollectProfile(ctx: Context, profile: LocalCollectProfile) {
        prefs(ctx).edit().putString(KEY_LOCAL_COLLECT_PROFILE, profile.toJson().toString()).apply()
    }

    private fun storageKey(patientId: String) = "$STORAGE_KEY_PREFIX$patientId"

    fun appendRecord(ctx: Context, patientId: String, payload: JSONObject) {
        val pid = patientId.trim()
        if (pid.isEmpty()) return
        val key = storageKey(pid)
        val raw = prefs(ctx).getString(key, null) ?: "[]"
        val arr = JSONArray(raw)
        val record = JSONObject(payload.toString())
        val now = System.currentTimeMillis()
        record.put("timestamp", now)
        record.put("dateTime", formatDateTimeZh(now))
        arr.put(record)
        val trimmed = trimEnd(arr, MAX_RECORDS)
        prefs(ctx).edit().putString(key, trimmed.toString()).apply()
    }

    fun getRecords(ctx: Context, patientId: String): JSONArray {
        val pid = patientId.trim()
        if (pid.isEmpty()) return JSONArray()
        val raw = prefs(ctx).getString(storageKey(pid), null) ?: return JSONArray()
        return try {
            JSONArray(raw)
        } catch (_: Exception) {
            JSONArray()
        }
    }

    /**
     * 取「最新」一条监测记录：优先 [timestamp] 最大；若无时间戳则取数组最后一条。
     */
    fun pickLatestRecord(records: JSONArray): JSONObject? {
        val n = records.length()
        if (n == 0) return null
        var maxTs = Long.MIN_VALUE
        for (i in 0 until n) {
            val o = records.optJSONObject(i) ?: continue
            if (o.has("timestamp") && !o.isNull("timestamp")) {
                val t = o.optLong("timestamp", Long.MIN_VALUE)
                if (t > maxTs) maxTs = t
            }
        }
        if (maxTs == Long.MIN_VALUE) return records.optJSONObject(n - 1)
        var lastIdx = -1
        for (i in 0 until n) {
            val o = records.optJSONObject(i) ?: continue
            if (o.has("timestamp") && !o.isNull("timestamp") && o.optLong("timestamp") == maxTs) {
                lastIdx = i
            }
        }
        return if (lastIdx >= 0) records.optJSONObject(lastIdx) else records.optJSONObject(n - 1)
    }

    private fun trimEnd(arr: JSONArray, max: Int): JSONArray {
        if (arr.length() <= max) return arr
        val out = JSONArray()
        val start = arr.length() - max
        for (i in start until arr.length()) {
            out.put(arr.get(i))
        }
        return out
    }

    private fun formatDateTimeZh(epoch: Long): String {
        val sdf = SimpleDateFormat("yyyy/MM/dd HH:mm:ss", Locale.CHINA)
        return sdf.format(Date(epoch))
    }
}
