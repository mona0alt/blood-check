package com.bloodcheck.app

import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File
import java.nio.charset.StandardCharsets
import java.nio.file.Files
import java.io.ByteArrayOutputStream
import java.util.zip.ZipFile
import java.util.zip.ZipInputStream

class PatientCsvWritersTest {
    @Test
    fun appendsRecordsHeaderOnceAndIncludesSpectrumIndexes() {
        val dir = tempDir("patient-csv")
        val file = File(dir, PatientDataFileStore.RECORDS_FILE)
        val writer = PatientRecordCsvWriter()
        val record = TestJsonObject(
            mapOf(
                "dateTime" to "2026/06/07 15:30:00",
                "patient_id" to "p001",
                "patient_info" to TestJsonObject(mapOf("姓名" to "张三", "年龄" to "40")),
                "hemoglobin" to TestJsonObject(mapOf("value" to 120.0, "unit" to "g/L"))
            )
        )

        writer.appendRecord(file, record, spectrumStartIndex = 10, spectrumEndIndex = 209)
        writer.appendRecord(file, record, spectrumStartIndex = 210, spectrumEndIndex = 409)

        val lines = file.readText(StandardCharsets.UTF_8).removePrefix("\uFEFF").lines().filter { it.isNotEmpty() }
        assertEquals(3, lines.size)
        assertTrue(lines[0].contains("\"光谱起始序号\""))
        assertTrue(lines[0].contains("\"光谱结束序号\""))
        assertTrue(lines[1].contains("\"10\",\"209\""))
        assertTrue(lines[2].contains("\"210\",\"409\""))
        assertEquals(1, lines.count { it.contains("\"光谱起始序号\"") })
    }

    @Test
    fun appendsSpectrumHeaderOnce() {
        val dir = tempDir("patient-csv")
        val file = File(dir, PatientDataFileStore.SPECTRUM_FILE)
        val writer = SpectrumCsvWriter()
        val rows = listOf(
            BleRawSample(0, "s1", 1000L, 1, 2, 3, 4.0, 5.0),
            BleRawSample(1, "s1", 1010L, 6, 7, 8, 9.0, 10.0)
        )

        writer.appendSamples(file, rows)
        writer.appendSamples(file, rows.map { it.copy(sampleIndex = it.sampleIndex + 2) })

        val lines = file.readText(StandardCharsets.UTF_8).removePrefix("\uFEFF").lines().filter { it.isNotEmpty() }
        assertEquals(5, lines.size)
        assertEquals("\"sample_index\",\"session_id\",\"采集时间\",\"C1\",\"C2\",\"C3\",\"红光\",\"红外\"", lines[0])
        assertTrue(lines[1].contains("\"0\",\"s1\""))
        assertTrue(lines[4].contains("\"3\",\"s1\""))
        assertEquals(1, lines.count { it.contains("\"sample_index\"") })
    }

    @Test
    fun monitorSpectrumNextSampleIndexContinuesAfterExistingCsvTail() {
        val dir = tempDir("patient-csv-tail")
        val file = File(dir, PatientDataFileStore.SPECTRUM_FILE)
        SpectrumCsvWriter().appendSamples(
            file,
            listOf(
                BleRawSample(40, "s1", 1000L, 1, 2, 3, 4.0, 5.0),
                BleRawSample(41, "s1", 1010L, 6, 7, 8, 9.0, 10.0)
            )
        )
        file.appendText("\n", StandardCharsets.UTF_8)

        assertEquals(42L, MonitorSpectrumFileIndex.nextSampleIndex(file))
    }

    @Test
    fun monitorFileIoFailurePolicyCoalescesToastWindow() {
        assertTrue(MonitorFileIoFailurePolicy.shouldShowToast(10_000L, 0L))
        assertFalse(MonitorFileIoFailurePolicy.shouldShowToast(15_000L, 10_000L))
        assertTrue(MonitorFileIoFailurePolicy.shouldShowToast(20_000L, 10_000L))
    }

    @Test
    fun zipExportIncludesRecordsAndSpectrumWhenPresent() {
        val dir = tempDir("patient-zip")
        File(dir, PatientDataFileStore.RECORDS_FILE).writeText("records")
        File(dir, PatientDataFileStore.SPECTRUM_FILE).writeText("spectrum")
        val zip = File(tempDir("patient-zip-out"), "out.zip")

        val result = PatientDataZipExporter().zipDataSet(dir, zip)

        assertTrue(result.created)
        assertEquals(
            listOf(PatientDataFileStore.RECORDS_FILE, PatientDataFileStore.SPECTRUM_FILE),
            result.includedFiles
        )
        assertTrue(result.missingFiles.isEmpty())
        ZipFile(zip).use { z ->
            assertTrue(z.getEntry(PatientDataFileStore.RECORDS_FILE) != null)
            assertTrue(z.getEntry(PatientDataFileStore.SPECTRUM_FILE) != null)
        }
    }

    @Test
    fun zipExportCanWriteToOutputStream() {
        val dir = tempDir("patient-zip-stream")
        File(dir, PatientDataFileStore.RECORDS_FILE).writeText("records")
        File(dir, PatientDataFileStore.SPECTRUM_FILE).writeText("spectrum")
        val out = ByteArrayOutputStream()

        val result = PatientDataZipExporter().zipDataSet(dir, out)

        assertTrue(result.created)
        assertEquals(
            listOf(PatientDataFileStore.RECORDS_FILE, PatientDataFileStore.SPECTRUM_FILE),
            zipEntryNames(out.toByteArray())
        )
    }

    @Test
    fun zipExportReportsMissingFilesWhenNoDatasetFilesExist() {
        val dir = tempDir("patient-zip-empty")
        val zip = File(tempDir("patient-zip-out"), "out.zip")

        val result = PatientDataZipExporter().zipDataSet(dir, zip)

        assertFalse(result.created)
        assertFalse(zip.exists())
        assertTrue(result.includedFiles.isEmpty())
        assertEquals(
            listOf(PatientDataFileStore.RECORDS_FILE, PatientDataFileStore.SPECTRUM_FILE),
            result.missingFiles
        )
    }

    private class TestJsonObject(private val values: Map<String, Any?>) : JSONObject() {
        override fun optJSONObject(name: String?): JSONObject? = values[name] as? JSONObject

        override fun optString(name: String?): String = values[name]?.toString().orEmpty()

        override fun isNull(name: String?): Boolean = !values.containsKey(name) || values[name] == null

        override fun optDouble(name: String?): Double {
            val value = values[name]
            return when (value) {
                is Number -> value.toDouble()
                is String -> value.toDoubleOrNull() ?: Double.NaN
                else -> Double.NaN
            }
        }
    }

    private fun tempDir(prefix: String): File = Files.createTempDirectory(prefix).toFile()

    private fun zipEntryNames(bytes: ByteArray): List<String> {
        val names = mutableListOf<String>()
        ZipInputStream(bytes.inputStream()).use { zip ->
            while (true) {
                val entry = zip.nextEntry ?: break
                names += entry.name
                zip.closeEntry()
            }
        }
        return names
    }
}
