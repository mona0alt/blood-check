package com.bloodcheck.app

import android.content.Context
import org.json.JSONArray
import org.json.JSONObject
import kotlin.math.roundToInt
import java.io.File
import java.io.FileOutputStream
import java.io.OutputStreamWriter
import java.nio.charset.StandardCharsets

/**
 * 对齐 [danbaijiance-uniapp/utils/excelExport.js] 中 formatPatientDataForExport + CSV BOM。
 */
object PatientDataExport {

    private val EXPORT_HEADERS = listOf(
        "时间",
        "患者ID",
        "统一ID",
        "住院号",
        "姓名",
        "性别",
        "年龄",
        "血红蛋白(Hb)",
        "血红蛋白单位",
        "血红蛋白状态",
        "乳酸(Lac)",
        "乳酸单位",
        "乳酸状态",
        "血流灌注指数(PI)",
        "PI分类",
        "PI说明",
        "血氧饱和度(SpO2)",
        "氧分压(PaO2)",
        "心率(HR)",
        "体温",
        "氧合血红蛋白分数"
    )

    fun formatRecordsForExport(records: JSONArray): List<Map<String, String>> {
        val list = mutableListOf<Map<String, String>>()
        for (i in 0 until records.length()) {
            val rec = records.optJSONObject(i) ?: continue
            list.add(formatOneRecord(rec))
        }
        return list
    }

    private fun formatOneRecord(record: JSONObject): Map<String, String> {
        val info = record.optJSONObject("patient_info")
        val hb = record.optJSONObject("hemoglobin")
        val lac = record.optJSONObject("lactate")
        val pi = record.optJSONObject("perfusion_index")

        fun infoStr(key: String): String = info?.optString(key).orEmpty()

        // patient_info 来自 Python 时键名为「氧合血红蛋白分数(FO2Hb)」，与 CRF 表头一致
        fun infoStrFo2Hb(): String {
            val a = infoStr("氧合血红蛋白分数")
            if (a.isNotBlank()) return a
            return infoStr("氧合血红蛋白分数(FO2Hb)")
        }

        val hbVal = when {
            hb != null && !hb.isNull("value") -> hb.optDouble("value").let { if (it.isNaN()) "" else it.toString() }
            infoStr("血红蛋白").isNotBlank() -> infoStr("血红蛋白")
            else -> ""
        }
        val lacVal = when {
            lac != null && !lac.isNull("value") -> lac.optDouble("value").let { if (it.isNaN()) "" else it.toString() }
            infoStr("乳酸").isNotBlank() -> infoStr("乳酸")
            else -> ""
        }
        val piVal = when {
            pi != null && !pi.isNull("value") -> pi.optDouble("value").let { if (it.isNaN()) "" else it.toString() }
            infoStr("血流灌注指数").isNotBlank() -> infoStr("血流灌注指数")
            else -> ""
        }

        fun paO2IntCsv(s: String): String {
            val t = s.trim()
            if (t.isEmpty()) return ""
            return t.toDoubleOrNull()?.roundToInt()?.toString() ?: t
        }

        return mapOf(
            "时间" to record.optString("dateTime"),
            "患者ID" to record.optString("patient_id"),
            "统一ID" to infoStr("统一ID"),
            "住院号" to infoStr("住院号"),
            "姓名" to infoStr("姓名"),
            "性别" to infoStr("性别"),
            "年龄" to infoStr("年龄"),
            "血红蛋白(Hb)" to hbVal,
            "血红蛋白单位" to (hb?.optString("unit").takeIf { !it.isNullOrBlank() } ?: "g/L"),
            "血红蛋白状态" to (hb?.optString("clinical_interpretation") ?: ""),
            "乳酸(Lac)" to lacVal,
            "乳酸单位" to (lac?.optString("unit").takeIf { !it.isNullOrBlank() } ?: "mmol/L"),
            "乳酸状态" to (lac?.optString("clinical_interpretation") ?: ""),
            "血流灌注指数(PI)" to piVal,
            "PI分类" to (pi?.optString("classification") ?: ""),
            "PI说明" to (pi?.optString("interpretation") ?: ""),
            "血氧饱和度(SpO2)" to infoStr("血氧饱和度"),
            "氧分压(PaO2)" to paO2IntCsv(infoStr("氧分压")),
            "心率(HR)" to infoStr("心率"),
            "体温" to infoStr("体温"),
            "氧合血红蛋白分数" to infoStrFo2Hb()
        )
    }

    /** 成功返回用于展示的完整保存路径，失败返回 null（由界面弹窗/Toast 提示） */
    fun exportCsvToDownloadsDir(context: Context, patientId: String, records: JSONArray): String? {
        val rows = formatRecordsForExport(records)
        if (rows.isEmpty()) return null
        val firstName = rows.firstOrNull()?.get("姓名").orEmpty().ifBlank { patientId }
        val fileName = "患者数据_${firstName}_${patientId}_${System.currentTimeMillis()}.csv"
        val dir = context.getExternalFilesDir(android.os.Environment.DIRECTORY_DOWNLOADS)
            ?: context.filesDir
        dir.mkdirs()
        val file = File(dir, fileName)
        val csv = buildCsv(rows)
        FileOutputStream(file).use { fos ->
            fos.write(0xEF)
            fos.write(0xBB)
            fos.write(0xBF)
            OutputStreamWriter(fos, StandardCharsets.UTF_8).use { w ->
                w.write(csv)
            }
        }
        val displayPath = file.absolutePath.replace("/storage/emulated/0/", "/内部存储/")
        return displayPath
    }

    private fun buildCsv(rows: List<Map<String, String>>): String {
        val sb = StringBuilder()
        sb.append(EXPORT_HEADERS.joinToString(",") { "\"$it\"" })
        sb.append('\n')
        for (row in rows) {
            val line = EXPORT_HEADERS.joinToString(",") { header ->
                val v = row[header] ?: ""
                val escaped = v.replace("\"", "\"\"")
                "\"$escaped\""
            }
            sb.append(line).append('\n')
        }
        return sb.toString()
    }
}
