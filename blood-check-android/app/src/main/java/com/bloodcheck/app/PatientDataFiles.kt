package com.bloodcheck.app

import android.content.Context
import android.os.Environment
import java.io.File

data class PatientDataSetSummary(
    val folderName: String,
    val dir: File,
    val fileCount: Int,
    val totalBytes: Long,
    val lastModifiedMillis: Long,
    val hasRecords: Boolean,
    val hasSpectrum: Boolean
) {
    val isComplete: Boolean get() = hasRecords && hasSpectrum
}

data class PatientDataDeleteResult(
    val requested: Int,
    val deleted: Int,
    val failed: Int,
    val skippedProtected: Int
)

class PatientDataFileStore(private val rootDir: File) {
    companion object {
        const val RECORDS_FILE = "records.csv"
        const val SPECTRUM_FILE = "spectrum.csv"

        fun fromContext(context: Context): PatientDataFileStore {
            val base = context.getExternalFilesDir(Environment.DIRECTORY_DOCUMENTS)
                ?: context.filesDir
            return PatientDataFileStore(File(base, "patient_data"))
        }

        fun safeName(raw: String): String {
            val cleaned = raw.trim()
                .replace(Regex("""[\\/:*?"<>|]+"""), "_")
                .replace(Regex("""\s+"""), "_")
                .trim('_')
            return cleaned.ifBlank { "unknown" }
        }
    }

    fun root(): File {
        rootDir.mkdirs()
        return rootDir
    }

    fun patientFolderName(patientName: String, patientId: String): String {
        return "${safeName(patientName)}_${safeName(patientId)}"
    }

    fun patientDirectory(patientName: String, patientId: String): File {
        return File(root(), patientFolderName(patientName, patientId))
    }

    fun recordsFile(dir: File): File = File(dir, RECORDS_FILE)

    fun spectrumFile(dir: File): File = File(dir, SPECTRUM_FILE)

    fun listDataSets(): List<PatientDataSetSummary> {
        val dirs = root().listFiles()?.filter { it.isDirectory }.orEmpty()
        return dirs.sortedBy { it.name }.map { dir ->
            val files = dir.listFiles()?.filter { it.isFile }.orEmpty()
            val records = recordsFile(dir)
            val spectrum = spectrumFile(dir)
            PatientDataSetSummary(
                folderName = dir.name,
                dir = dir,
                fileCount = files.size,
                totalBytes = files.sumOf { it.length() },
                lastModifiedMillis = files.maxOfOrNull { it.lastModified() } ?: dir.lastModified(),
                hasRecords = records.isFile,
                hasSpectrum = spectrum.isFile
            )
        }
    }

    fun deleteDataSets(folderNames: List<String>, protectedFolderName: String?): PatientDataDeleteResult {
        var deleted = 0
        var failed = 0
        var skipped = 0
        folderNames.distinct().forEach { folder ->
            if (protectedFolderName != null && folder == protectedFolderName) {
                skipped += 1
                return@forEach
            }
            val dir = dataSetDirectoryForDelete(folder)
            if (dir == null) {
                failed += 1
                return@forEach
            }
            if (!dir.exists()) {
                deleted += 1
            } else if (dir.isDirectory && dir.deleteRecursively()) {
                deleted += 1
            } else {
                failed += 1
            }
        }
        return PatientDataDeleteResult(
            requested = folderNames.distinct().size,
            deleted = deleted,
            failed = failed,
            skippedProtected = skipped
        )
    }

    private fun dataSetDirectoryForDelete(folderName: String): File? {
        if (folderName.isBlank() || folderName == "." || folderName == "..") {
            return null
        }
        if (folderName.contains('/') || folderName.contains('\\')) {
            return null
        }

        val canonicalRoot = root().canonicalFile
        val target = File(canonicalRoot, folderName).canonicalFile
        return if (target.parentFile == canonicalRoot && target.name == folderName) {
            target
        } else {
            null
        }
    }
}
