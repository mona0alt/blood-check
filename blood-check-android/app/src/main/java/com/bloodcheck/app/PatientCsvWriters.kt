package com.bloodcheck.app

import org.json.JSONObject
import java.io.File
import java.io.FileOutputStream
import java.io.OutputStream
import java.io.OutputStreamWriter
import java.nio.charset.StandardCharsets
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.zip.ZipEntry
import java.util.zip.ZipOutputStream

data class BleRawSample(
    val sampleIndex: Long,
    val sessionId: String,
    val capturedAtMillis: Long,
    val c1: Int,
    val c2: Int,
    val c3: Int,
    val red: Double,
    val infrared: Double
)

data class PatientZipResult(
    val created: Boolean,
    val includedFiles: List<String>,
    val missingFiles: List<String>
)

object CsvText {
    fun cell(value: Any?): String {
        val escaped = value?.toString().orEmpty().replace("\"", "\"\"")
        return "\"$escaped\""
    }

    fun line(values: List<Any?>): String = values.joinToString(",") { cell(it) }

    fun ensureParent(file: File) {
        file.parentFile?.mkdirs()
    }

    fun appendUtf8BomCsv(file: File, header: String, line: String) {
        ensureParent(file)
        val isNew = !file.exists() || file.length() == 0L
        FileOutputStream(file, true).use { fos ->
            if (isNew) {
                fos.write(0xEF)
                fos.write(0xBB)
                fos.write(0xBF)
            }
            OutputStreamWriter(fos, StandardCharsets.UTF_8).use { writer ->
                if (isNew) writer.write(header + "\n")
                writer.write(line + "\n")
            }
        }
    }
}

class PatientRecordCsvWriter {
    private val headers = PatientDataExport.EXPORT_HEADERS + listOf("光谱起始序号", "光谱结束序号")
    private val headerLine = CsvText.line(headers)

    fun appendRecord(file: File, record: JSONObject, spectrumStartIndex: Long, spectrumEndIndex: Long) {
        val row = PatientDataExport.formatOneRecordForExport(record).toMutableMap()
        row["光谱起始序号"] = spectrumStartIndex.toString()
        row["光谱结束序号"] = spectrumEndIndex.toString()

        CsvText.appendUtf8BomCsv(
            file = file,
            header = headerLine,
            line = CsvText.line(headers.map { row[it] ?: "" })
        )
    }
}

class SpectrumCsvWriter {
    private val headers = listOf("sample_index", "session_id", "采集时间", "C1", "C2", "C3", "红光", "红外")
    private val headerLine = CsvText.line(headers)
    private val dateFormat = SimpleDateFormat("yyyy/MM/dd HH:mm:ss.SSS", Locale.CHINA)

    fun appendSamples(file: File, samples: List<BleRawSample>) {
        if (samples.isEmpty()) return

        CsvText.ensureParent(file)
        val isNew = !file.exists() || file.length() == 0L
        FileOutputStream(file, true).use { fos ->
            if (isNew) {
                fos.write(0xEF)
                fos.write(0xBB)
                fos.write(0xBF)
            }
            OutputStreamWriter(fos, StandardCharsets.UTF_8).use { writer ->
                if (isNew) writer.write(headerLine + "\n")
                samples.forEach { sample ->
                    writer.write(
                        CsvText.line(
                            listOf(
                                sample.sampleIndex,
                                sample.sessionId,
                                dateFormat.format(Date(sample.capturedAtMillis)),
                                sample.c1,
                                sample.c2,
                                sample.c3,
                                sample.red,
                                sample.infrared
                            )
                        ) + "\n"
                    )
                }
            }
        }
    }
}

class PatientDataZipExporter {
    fun zipDataSet(patientDir: File, outputZip: File): PatientZipResult {
        outputZip.parentFile?.mkdirs()
        FileOutputStream(outputZip).use { output ->
            return zipDataSet(patientDir, output)
        }
    }

    fun zipDataSet(patientDir: File, output: OutputStream): PatientZipResult {
        val expected = listOf(PatientDataFileStore.RECORDS_FILE, PatientDataFileStore.SPECTRUM_FILE)
        val existing = expected
            .map { File(patientDir, it) }
            .filter { it.isFile }

        if (existing.isEmpty()) {
            return PatientZipResult(
                created = false,
                includedFiles = emptyList(),
                missingFiles = expected
            )
        }

        ZipOutputStream(output).use { zip ->
            existing.forEach { file ->
                zip.putNextEntry(ZipEntry(file.name))
                file.inputStream().use { input -> input.copyTo(zip) }
                zip.closeEntry()
            }
        }

        return PatientZipResult(
            created = true,
            includedFiles = existing.map { it.name },
            missingFiles = expected.filterNot { name -> existing.any { it.name == name } }
        )
    }
}
